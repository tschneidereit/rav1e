#!/bin/bash
# Run WASM encode/decode tests locally
# This script mirrors the wasi-decode-test CI job

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() { echo -e "${GREEN}==>${NC} $*"; }
warn() { echo -e "${YELLOW}==> WARNING:${NC} $*"; }
error() { echo -e "${RED}==> ERROR:${NC} $*"; exit 1; }

# Check dependencies
check_deps() {
    info "Checking dependencies..."
    
    if ! command -v wasmtime &> /dev/null; then
        error "wasmtime not found. Install with: curl https://wasmtime.dev/install.sh -sSf | bash"
    fi
    
    if ! command -v dav1d &> /dev/null; then
        error "dav1d not found. Install with: sudo apt install dav1d (or brew install dav1d)"
    fi
    
    if ! command -v cargo &> /dev/null; then
        error "cargo not found. Install Rust from https://rustup.rs"
    fi
    
    if ! rustup target list --installed | grep -q wasm32-wasip2; then
        warn "wasm32-wasip2 target not installed. Installing..."
        rustup target add wasm32-wasip2
    fi
}

# Build rav1e.wasm
build_wasm() {
    info "Building rav1e for wasm32-wasip2..."
    
    RUSTFLAGS="-C target-feature=+simd128,+relaxed-simd" cargo build \
        --target wasm32-wasip2 \
        --release \
        --no-default-features \
        --features binaries
    
    WASM_PATH="$PROJECT_ROOT/target/wasm32-wasip2/release/rav1e.wasm"
    if [[ ! -f "$WASM_PATH" ]]; then
        error "Build failed: $WASM_PATH not found"
    fi
    
    info "Built: $WASM_PATH ($(du -h "$WASM_PATH" | cut -f1))"
}

# Run tests
run_tests() {
    local test_filter="${1:-}"
    local extra_args=("${@:2}")
    
    info "Running WASM encode/decode tests..."
    
    local cmd=(
        cargo test
        --release
        --features wasm_decode_test
        --test wasm_encode_decode
    )
    
    if [[ -n "$test_filter" ]]; then
        cmd+=("$test_filter")
    fi
    
    cmd+=(-- --test-threads=1)
    
    if [[ ${#extra_args[@]} -gt 0 ]]; then
        cmd+=("${extra_args[@]}")
    fi
    
    "${cmd[@]}"
}

# Show usage
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [TEST_FILTER] [-- EXTRA_ARGS]

Run WASM encode/decode tests for rav1e.

Options:
    -h, --help          Show this help message
    -b, --build-only    Only build rav1e.wasm, don't run tests
    -t, --test-only     Only run tests (skip building)
    -i, --include-ignored, --slow
                        Include ignored (slow) tests in test run
    -v, --verbose       Show test output (--nocapture)

Examples:
    $(basename "$0")                        # Run all non-ignored tests
    $(basename "$0") --slow                 # Run all tests including slow ones
    $(basename "$0") speed_tests            # Run only speed tests
    $(basename "$0") dimension_64x64        # Run a specific test
    $(basename "$0") -v quantizer           # Run quantizer tests with output
    $(basename "$0") -- --ignored           # Run only ignored tests

Test modules:
    speed_tests           Speed 0-10 tests (6 fast, 5 slow/ignored)
    dimension_tests       Various dimension tests (10 fast, 2 large/ignored)
    quantizer_tests       Quantizer 60/80/100/120 tests
    keyframe_tests        Keyframe and reordering tests
    bit_depth_tests       10-bit and 12-bit tests (ignored - requires dav1d HBD)
    chroma_sampling_tests 420/422/444 tests (ignored - requires dav1d validation)
    tile_tests            Tiled encoding tests
    still_picture_tests   Still picture mode tests
    binary_tests          CLI tests: 1-pass/2-pass/3-pass, QP/bitrate modes

Test counts (default run):
    34 tests run, 13 ignored (slow/HBD), for 100% coverage parity with native
EOF
}

# Main
main() {
    local build_only=false
    local test_only=false
    local include_ignored=false
    local verbose=false
    local test_filter=""
    local extra_args=()
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                usage
                exit 0
                ;;
            -b|--build-only)
                build_only=true
                shift
                ;;
            -t|--test-only)
                test_only=true
                shift
                ;;
            -i|--include-ignored|--slow)
                include_ignored=true
                shift
                ;;
            -v|--verbose)
                verbose=true
                shift
                ;;
            --)
                shift
                extra_args+=("$@")
                break
                ;;
            -*)
                error "Unknown option: $1"
                ;;
            *)
                test_filter="$1"
                shift
                ;;
        esac
    done
    
    check_deps
    
    if [[ "$test_only" != true ]]; then
        build_wasm
    fi
    
    if [[ "$build_only" != true ]]; then
        if [[ "$include_ignored" == true ]]; then
            extra_args+=(--include-ignored)
        fi
        if [[ "$verbose" == true ]]; then
            extra_args+=(--nocapture)
        fi
        run_tests "$test_filter" "${extra_args[@]}"
    fi
    
    info "Done!"
}

main "$@"
