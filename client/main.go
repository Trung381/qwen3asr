// Qwen3-ASR gRPC Client — Go
//
// Streams microphone audio (via ffmpeg) or a WAV file to the Qwen3-ASR gRPC
// server and prints partial/final transcriptions in real-time.
//
// REQUIREMENTS:
//
//	ffmpeg must be installed and on PATH.
//	On Windows: https://ffmpeg.org/download.html (add to PATH)
//	On Linux:   sudo apt-get install -y ffmpeg
//
// USAGE (from Windows PowerShell):
//
//	# Stream mic:
//	go run . --language English
//
//	# List available audio input devices (Windows):
//	go run . --list-devices
//
//	# Use a specific device:
//	go run . --device "Microphone (Realtek Audio)" --language English
//
//	# Stream a WAV file:
//	go run . --file ..\test_en.wav --language English
//
// BUILD (produces asr-client.exe on Windows):
//
//	go build -o asr-client.exe .
package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/smartiva/qwen3-asr-client/proto"
)

// ─── VAD (Voice Activity Detection) config ────────────────────────────────────

const (
	// silenceThreshold: RMS energy below this = silence.
	// Tune this for your mic: louder room → raise to 0.01 or 0.02.
	silenceThreshold = 0.005

	// silenceChunks: how many consecutive silent chunks before we treat it
	// as end-of-speech. At 500ms chunks: 3 × 500ms = 1.5 seconds of silence.
	silenceChunks = 2

	// minSpeechChunks: minimum chunks of speech before silence detection kicks in.
	// Prevents triggering on the very first chunk if mic is quiet.
	minSpeechChunks = 2
)

const sampleRate = 16000

func main() {
	host := flag.String("host", "localhost", "gRPC server host")
	port := flag.Int("port", 50051, "gRPC server port")
	language := flag.String("language", "", "Language hint (e.g. 'English', 'Chinese'). Empty = auto-detect")
	chunkMs := flag.Int("chunk-ms", 500, "Audio chunk size in milliseconds")
	fileMode := flag.String("file", "", "Path to audio file to stream instead of microphone")
	device := flag.String("device", "", "Audio input device name (Windows: from --list-devices, Linux: e.g. 'default')")
	listDevices := flag.Bool("list-devices", false, "List available audio input devices and exit")
	flag.Parse()

	if *listDevices {
		listAudioDevices()
		return
	}

	addr := fmt.Sprintf("%s:%d", *host, *port)
	fmt.Printf("Connecting to gRPC server at %s...\n", addr)

	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}
	defer conn.Close()

	client := pb.NewASRServiceClient(conn)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigs
		fmt.Println("\n[interrupted]")
		cancel()
	}()

	// onFinal is called whenever a complete utterance is transcribed.
	// Replace this with your LLM call, e.g. sendToLLM(text).
	onFinal := func(lang, text string) {
		fmt.Printf("\n>>> [ASR FINAL] lang=%q\n    text: %s\n\n", lang, text)
		// TODO: send `text` to your LLM module here, e.g.:
		// go sendToLLM(text)
	}

	if *fileMode != "" {
		streamFile(ctx, client, *fileMode, *language, *chunkMs, onFinal)
	} else {
		streamMic(ctx, client, *language, *chunkMs, *device, onFinal)
	}
}

// ─── File streaming ──────────────────────────────────────────────────────────

func streamFile(ctx context.Context, client pb.ASRServiceClient, filePath, language string, chunkMs int, onFinal func(lang, text string)) {
	fmt.Printf("Reading audio file via ffmpeg: %s\n", filePath)

	// Use ffmpeg to decode any audio format to raw float32 PCM at 16 kHz mono
	cmd := exec.CommandContext(ctx, "ffmpeg",
		"-i", filePath,
		"-ar", fmt.Sprintf("%d", sampleRate),
		"-ac", "1",
		"-f", "f32le",
		"-",
	)
	cmd.Stderr = nil // suppress ffmpeg banner
	pipe, err := cmd.StdoutPipe()
	if err != nil {
		log.Fatalf("ffmpeg pipe error: %v", err)
	}
	if err := cmd.Start(); err != nil {
		log.Fatalf("ffmpeg start error: %v\nMake sure ffmpeg is installed and on PATH.", err)
	}

	stream, err := client.TranscribeStream(ctx)
	if err != nil {
		log.Fatalf("TranscribeStream failed: %v", err)
	}

	// File mode: no VAD needed, ffmpeg closes pipe at EOF → is_final triggered naturally
	sendPCMStream(pipe, stream, language, chunkMs, false, onFinal)
	cmd.Wait()
}

// ─── Mic streaming via ffmpeg ─────────────────────────────────────────────────

func streamMic(ctx context.Context, client pb.ASRServiceClient, language string, chunkMs int, device string, onFinal func(lang, text string)) {
	args := buildFFmpegMicArgs(device)
	fmt.Printf("Opening microphone via ffmpeg...\n")
	fmt.Printf("  Platform: %s | Device: %q | Chunk: %dms | Language: %q\n\n", runtime.GOOS, device, chunkMs, language)
	fmt.Printf("  Silence threshold: %.4f RMS | Silence timeout: %d × %dms = %dms\n",
		silenceThreshold, silenceChunks, chunkMs, silenceChunks*chunkMs)
	fmt.Println("  Speak now (Ctrl+C to stop):\n")

	cmd := exec.CommandContext(ctx, "ffmpeg", args...)
	cmd.Stderr = os.Stderr // show ffmpeg errors (wrong device name etc.)
	pipe, err := cmd.StdoutPipe()
	if err != nil {
		log.Fatalf("ffmpeg pipe error: %v", err)
	}
	if err := cmd.Start(); err != nil {
		log.Fatalf("Could not start ffmpeg: %v\n\nMake sure ffmpeg is installed:\n  Windows: winget install Gyan.FFmpeg\n  Linux:   sudo apt-get install ffmpeg", err)
	}

	// Mic mode: loop forever, restarting a fresh gRPC stream after each utterance.
	// VAD detects end-of-speech → sends is_final=true → awaits [final] → calls onFinal → repeat.
	for {
		stream, err := client.TranscribeStream(ctx)
		if err != nil {
			if ctx.Err() != nil {
				return // user pressed Ctrl+C
			}
			log.Fatalf("TranscribeStream failed: %v", err)
		}

		sendPCMStream(pipe, stream, language, chunkMs, true, onFinal)

		if ctx.Err() != nil {
			return // interrupted
		}
		fmt.Println("  [Listening for next utterance...]")
	}
}

// buildFFmpegMicArgs returns platform-specific ffmpeg arguments for mic capture.
func buildFFmpegMicArgs(device string) []string {
	base := []string{
		"-ar", fmt.Sprintf("%d", sampleRate),
		"-ac", "1",
		"-f", "f32le",
		"pipe:1",
	}

	switch runtime.GOOS {
	case "windows":
		dev := device
		if dev == "" {
			// ffmpeg -f dshow -i audio="default" (picks first available)
			dev = "audio=default"
		} else {
			dev = "audio=" + device
		}
		return append([]string{"-f", "dshow", "-i", dev}, base...)
	case "darwin":
		dev := device
		if dev == "" {
			dev = ":0"
		}
		return append([]string{"-f", "avfoundation", "-i", dev}, base...)
	default: // Linux
		dev := device
		if dev == "" {
			dev = "default"
		}
		return append([]string{"-f", "pulse", "-i", dev}, base...)
	}
}

// ─── PCM stream sender ────────────────────────────────────────────────────────

// rmsEnergy computes root-mean-square energy of a float32 LE PCM byte slice.
func rmsEnergy(raw []byte) float64 {
	n := len(raw) / 4
	if n == 0 {
		return 0
	}
	var sum float64
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint32(raw[i*4 : i*4+4])
		s := float64(math.Float32frombits(bits))
		sum += s * s
	}
	return math.Sqrt(sum / float64(n))
}

// sendPCMStream reads raw float32 PCM from r and streams it to the gRPC server.
// If useVAD=true, silence detection is active: after silenceChunks consecutive
// silent chunks (post minSpeechChunks of speech), is_final=true is sent and
// the function returns so the caller can start a fresh gRPC stream.
func sendPCMStream(r io.Reader, stream pb.ASRService_TranscribeStreamClient, language string, chunkMs int, useVAD bool, onFinal func(lang, text string)) {
	chunkSamples := sampleRate * chunkMs / 1000
	chunkBytes := chunkSamples * 4 // float32 = 4 bytes
	buf := make([]byte, chunkBytes)
	reader := bufio.NewReaderSize(r, chunkBytes*4)

	doneSend := make(chan struct{})

	go func() {
		defer close(doneSend)
		silentCount := 0
		speechCount := 0

		for {
			n, err := io.ReadFull(reader, buf)
			if n > 0 {
				isFinal := err != nil // EOF = last chunk of file

				if useVAD && !isFinal {
					energy := rmsEnergy(buf[:n])
					if energy < silenceThreshold {
						silentCount++
					} else {
						silentCount = 0
						speechCount++
						fmt.Printf("  [speech] rms=%.4f\r", energy)
					}

					// End-of-speech: enough speech was heard, then silence long enough
					if speechCount >= minSpeechChunks && silentCount >= silenceChunks {
						fmt.Printf("\n  [silence detected — %.1fs — finalizing...]\n",
							float64(silentCount*chunkMs)/1000.0)
						isFinal = true
					}
				}

				chunk := &pb.AudioChunk{
					AudioData:  append([]byte(nil), buf[:n]...),
					SampleRate: sampleRate,
					Language:   language,
					IsFinal:    isFinal,
				}
				if sendErr := stream.Send(chunk); sendErr != nil {
					log.Printf("Send error: %v", sendErr)
					return
				}

				if isFinal {
					stream.CloseSend()
					return
				}
			}
			if err != nil {
				stream.CloseSend()
				return
			}
		}
	}()

	receiveResponses(stream, onFinal)
	<-doneSend // wait for sender goroutine to finish too
}

// ─── Response handler ─────────────────────────────────────────────────────────

func receiveResponses(stream pb.ASRService_TranscribeStreamClient, onFinal func(lang, text string)) {
	for {
		resp, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			// Suppress context-cancelled errors on Ctrl+C
			if ctx := stream.Context(); ctx.Err() != nil {
				break
			}
			log.Printf("Recv error: %v", err)
			break
		}
		if resp.IsFinal {
			fmt.Printf("\r[final]   lang=%q  text=%q\n", resp.Language, resp.Text)
			if onFinal != nil && resp.Text != "" {
				onFinal(resp.Language, resp.Text)
			}
		} else {
			// Note: If the server is configured with ASR_MODE=non-streaming,
			// it buffers audio and does not send partial results until the end.
			fmt.Printf("\r[partial] %q                    ", resp.Text)
		}
	}
}

// ─── Device listing ───────────────────────────────────────────────────────────

func listAudioDevices() {
	var args []string
	switch runtime.GOOS {
	case "windows":
		fmt.Println("Listing Windows DirectShow audio devices:\n")
		args = []string{"-list_devices", "true", "-f", "dshow", "-i", "dummy"}
	case "darwin":
		fmt.Println("Listing macOS AVFoundation audio devices:\n")
		args = []string{"-f", "avfoundation", "-list_devices", "true", "-i", ""}
	default:
		fmt.Println("Linux: use 'pactl list sources short' to list PulseAudio input devices, or 'arecord -l' for ALSA.\n")
		return
	}

	cmd := exec.Command("ffmpeg", args...)
	// ffmpeg writes device list to stderr — combine into stdout so it shows
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	cmd.Run() // exit code 1 is expected; ignore
	fmt.Println("\nUse the device name with: go run . --device \"Device Name Here\"")
}

// ─── File WAV reader (fallback, float32 PCM) ─────────────────────────────────

func readWavMono16k(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var header [44]byte
	if _, err := io.ReadFull(f, header[:]); err != nil {
		return nil, fmt.Errorf("reading WAV header: %w", err)
	}

	fileSR := int(binary.LittleEndian.Uint32(header[24:28]))
	bitDepth := int(binary.LittleEndian.Uint16(header[34:36]))
	numChannels := int(binary.LittleEndian.Uint16(header[22:24]))

	data, err := io.ReadAll(f)
	if err != nil {
		return nil, err
	}

	var samples []float32
	switch bitDepth {
	case 16:
		for i := 0; i+1 < len(data); i += 2 * numChannels {
			s := float32(int16(binary.LittleEndian.Uint16(data[i:i+2]))) / 32768.0
			samples = append(samples, s)
		}
	case 32:
		for i := 0; i+3 < len(data); i += 4 * numChannels {
			bits := binary.LittleEndian.Uint32(data[i : i+4])
			samples = append(samples, math.Float32frombits(bits))
		}
	default:
		return nil, fmt.Errorf("unsupported bit depth: %d", bitDepth)
	}

	if fileSR != sampleRate {
		samples = resampleLinear(samples, fileSR, sampleRate)
	}
	return samples, nil
}

func resampleLinear(src []float32, srcRate, dstRate int) []float32 {
	ratio := float64(srcRate) / float64(dstRate)
	dstLen := int(float64(len(src)) / ratio)
	dst := make([]float32, dstLen)
	for i := range dst {
		pos := float64(i) * ratio
		lo := int(pos)
		hi := lo + 1
		if hi >= len(src) {
			hi = len(src) - 1
		}
		frac := float32(pos - float64(lo))
		dst[i] = src[lo]*(1-frac) + src[hi]*frac
	}
	return dst
}

var _ = time.Second
