.PHONY: proto client build-client run-client

# ─── Proto ───────────────────────────────────────────────────────────────────
# Generate Go gRPC stubs from proto/asr.proto into client/proto/
# Requires: protoc, protoc-gen-go, protoc-gen-go-grpc
# Install with:
#   go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
#   go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
proto:
	@echo "Generating Go proto stubs..."
	@mkdir -p client/proto
	protoc \
		--go_out=client/proto \
		--go_opt=paths=source_relative \
		--go-grpc_out=client/proto \
		--go-grpc_opt=paths=source_relative \
		-I proto \
		proto/asr.proto
	@echo "Done: client/proto/asr.pb.go, client/proto/asr_grpc.pb.go"

# ─── Client ──────────────────────────────────────────────────────────────────
build-client:
	@echo "Building Go client..."
	cd client && go build -o ../bin/asr-client .
	@echo "Binary: bin/asr-client"

# Run file-based smoke test (streams test_en.wav)
test-client:
	@echo "Running smoke test with test_en.wav..."
	cd client && go run . --file ../test_en.wav --language English

# Run live mic streaming
run-client:
	cd client && go run . --language English
