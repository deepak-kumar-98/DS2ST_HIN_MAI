# #!/usr/bin/env bash
# # =============================================================================
# #  run_asr_evaluation.sh
# #  Installs dependencies and runs the full Hindi ASR evaluation pipeline.
# #
# #  Usage:
# #    chmod +x run_asr_evaluation.sh
# #    ./run_asr_evaluation.sh                     # run all 4 models
# #    ./run_asr_evaluation.sh --models whisper_large_v3  # single model
# #    ./run_asr_evaluation.sh --max_utts 50        # quick smoke-test
# # =============================================================================

# set -euo pipefail

# # ─── Paths (edit if needed) ───────────────────────────────────────────────────
# AUDIO_DIR="/mnt/storage/aditya/Evaluation/hi_tts_outputs"
# TEXT_FILE="/mnt/storage/aditya/Evaluation/Benchmarking_dataset/Hindi_test_set.txt"
# OUTPUT_DIR="./asr_results"
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PYTHON_SCRIPT="${SCRIPT_DIR}/run_asr_evaluation.py"

# # ─── Colour helpers ──────────────────────────────────────────────────────────
# GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
# info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
# warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
# error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# # ─── Sanity checks ───────────────────────────────────────────────────────────
# info "=== Hindi ASR Evaluation Pipeline ==="
# info "Audio dir : ${AUDIO_DIR}"
# info "Text file : ${TEXT_FILE}"
# info "Output    : ${OUTPUT_DIR}"

# [[ -d "${AUDIO_DIR}"     ]] || error "Audio directory not found: ${AUDIO_DIR}"
# [[ -f "${TEXT_FILE}"     ]] || error "Text file not found: ${TEXT_FILE}"
# [[ -f "${PYTHON_SCRIPT}" ]] || error "Python script not found: ${PYTHON_SCRIPT}"

# NUM_WAV=$(find "${AUDIO_DIR}" -maxdepth 1 \( -name "*.wav" -o -name "*.flac" \) | wc -l)
# NUM_REF=$(grep -c . "${TEXT_FILE}" || true)
# info "Found ${NUM_WAV} audio files and ${NUM_REF} reference lines."

# # ─── Python environment ───────────────────────────────────────────────────────
# PYTHON=$(command -v python3 || command -v python || error "python3 not found")
# info "Using Python: $($PYTHON --version)"

# # Activate conda/venv if present alongside this script
# if [[ -f "${SCRIPT_DIR}/venv/bin/activate" ]]; then
#     info "Activating local venv …"
#     # shellcheck disable=SC1091
#     source "${SCRIPT_DIR}/venv/bin/activate"
# elif [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
#     info "Conda env active: ${CONDA_DEFAULT_ENV}"
# fi

# # ─── Install Python dependencies ─────────────────────────────────────────────
# info "Installing / verifying Python dependencies …"

# pip install --quiet --upgrade pip

# # Core ML
# pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
#     || pip install --quiet torch torchvision torchaudio   # CPU fallback

# # HuggingFace stack
# pip install --quiet \
#     transformers \
#     datasets \
#     accelerate \
#     sentencepiece \
#     tokenizers

# # Audio
# pip install --quiet \
#     librosa \
#     soundfile \
#     audioread

# # Metrics
# pip install --quiet jiwer

# # Whisper tokenizer extras
# pip install --quiet openai-whisper 2>/dev/null || true

# # Optional: NeMo for indic-conformer (large install — skip with --no-nemo)
# if [[ "${1:-}" != "--no-nemo" ]]; then
#     info "Attempting NeMo install (for indic-conformer-600m) …"
#     pip install --quiet \
#         "nemo_toolkit[asr]" \
#         2>/dev/null || warn "NeMo install failed – indic-conformer will use HF pipeline fallback."
# else
#     warn "--no-nemo passed; skipping NeMo install."
# fi

# # ─── GPU / CUDA status ───────────────────────────────────────────────────────
# $PYTHON - <<'PYEOF'
# import torch
# if torch.cuda.is_available():
#     print(f"[INFO]  CUDA available → {torch.cuda.get_device_name(0)} "
#           f"({torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)")
# else:
#     print("[WARN]  No CUDA detected — running on CPU (will be slow)")
# PYEOF

# # ─── Create output directory ─────────────────────────────────────────────────
# mkdir -p "${OUTPUT_DIR}"

# # ─── Build extra arguments from script args ──────────────────────────────────
# EXTRA_ARGS=""
# for arg in "$@"; do
#     EXTRA_ARGS="${EXTRA_ARGS} ${arg}"
# done

# # ─── Run evaluation ──────────────────────────────────────────────────────────
# info "Launching ASR evaluation …"
# START_TS=$(date +%s)

# $PYTHON "${PYTHON_SCRIPT}" \
#     --audio_dir  "${AUDIO_DIR}"  \
#     --text_file  "${TEXT_FILE}"  \
#     --output_dir "${OUTPUT_DIR}" \
#     ${EXTRA_ARGS}

# END_TS=$(date +%s)
# ELAPSED=$(( END_TS - START_TS ))
# info "Evaluation finished in ${ELAPSED}s"

# # ─── Print final report ───────────────────────────────────────────────────────
# REPORT="${OUTPUT_DIR}/asr_evaluation_report.txt"
# if [[ -f "${REPORT}" ]]; then
#     echo ""
#     echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#     cat "${REPORT}"
#     echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#     info "Full results saved to: ${OUTPUT_DIR}/"
#     info "Files:"
#     ls -lh "${OUTPUT_DIR}/"
# fi



#!/usr/bin/env bash
# =============================================================================
#  run_asr_evaluation.sh
#  Installs dependencies and runs the full Hindi ASR evaluation pipeline.
#
#  Usage:
#    chmod +x run_asr_evaluation.sh
#    ./run_asr_evaluation.sh                     # run all 4 models
#    ./run_asr_evaluation.sh --models whisper_large_v3  # single model
#    ./run_asr_evaluation.sh --max_utts 50        # quick smoke-test
# =============================================================================

# set -euo pipefail

# # ─── Paths (edit if needed) ───────────────────────────────────────────────────
# AUDIO_DIR="/mnt/storage/aditya/Evaluation/hi_tts_outputs"
# TEXT_FILE="/mnt/storage/aditya/Evaluation/Benchmarking_dataset/Hindi_test_set.txt"
# OUTPUT_DIR="./asr_results"
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PYTHON_SCRIPT="${SCRIPT_DIR}/run_asr_evaluation.py"

# # ─── Colour helpers ──────────────────────────────────────────────────────────
# GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
# info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
# warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
# error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# # ─── Sanity checks ───────────────────────────────────────────────────────────
# info "=== Hindi ASR Evaluation Pipeline ==="
# info "Audio dir : ${AUDIO_DIR}"
# info "Text file : ${TEXT_FILE}"
# info "Output    : ${OUTPUT_DIR}"

# [[ -d "${AUDIO_DIR}"     ]] || error "Audio directory not found: ${AUDIO_DIR}"
# [[ -f "${TEXT_FILE}"     ]] || error "Text file not found: ${TEXT_FILE}"
# [[ -f "${PYTHON_SCRIPT}" ]] || error "Python script not found: ${PYTHON_SCRIPT}"

# NUM_WAV=$(find "${AUDIO_DIR}" -maxdepth 1 \( -name "*.wav" -o -name "*.flac" \) | wc -l)
# NUM_REF=$(grep -c . "${TEXT_FILE}" || true)
# info "Found ${NUM_WAV} audio files and ${NUM_REF} reference lines."

# # ─── Python environment ───────────────────────────────────────────────────────
# PYTHON=$(command -v python3 || command -v python || error "python3 not found")
# info "Using Python: $($PYTHON --version)"

# # Activate conda/venv if present alongside this script
# if [[ -f "${SCRIPT_DIR}/venv/bin/activate" ]]; then
#     info "Activating local venv …"
#     # shellcheck disable=SC1091
#     source "${SCRIPT_DIR}/venv/bin/activate"
# elif [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
#     info "Conda env active: ${CONDA_DEFAULT_ENV}"
# fi

# # ─── Install Python dependencies ─────────────────────────────────────────────
# info "Installing / verifying Python dependencies …"

# # Suppress the harmless omegaconf PyYAML .* version-specifier warning that
# # appears with older pip resolvers. We capture stderr, filter that one line,
# # and re-print everything else so real errors are still visible.
# pip_install() {
#     local out
#     # Run pip; filter only the specific omegaconf PyYAML warning line
#     out=$(pip install --no-warn-script-location "$@" 2>&1) || {
#         # Non-zero exit: print filtered output then propagate failure
#         echo "${out}" | grep -v "Error parsing dependencies of omegaconf" \
#                       | grep -v "PyYAML (>=5.1.\*)" >&2
#         return 1
#     }
#     # Zero exit: print filtered output (suppress the benign warning)
#     echo "${out}" | grep -v "Error parsing dependencies of omegaconf" \
#                   | grep -v "PyYAML (>=5.1.\*)" \
#                   | grep -v "^$" \
#                   | grep -v "^WARNING: " \
#                   || true
# }

# # Upgrade pip itself quietly
# pip install --quiet --upgrade pip 2>/dev/null || true

# # ── Detect CUDA version for correct PyTorch wheel ────────────────────────────
# CUDA_VER=$(python3 -c "
# import subprocess, re
# try:
#     out = subprocess.check_output(['nvcc','--version'], stderr=subprocess.DEVNULL).decode()
#     m = re.search(r'release (\d+\.\d+)', out)
#     if m:
#         major, minor = m.group(1).split('.')
#         print(f'cu{major}{minor}')
#     else:
#         print('cpu')
# except Exception:
#     print('cpu')
# " 2>/dev/null || echo "cpu")

# info "Detected compute target: ${CUDA_VER}"

# # Map detected CUDA to a supported PyTorch index
# case "${CUDA_VER}" in
#     cu12*) PT_INDEX="https://download.pytorch.org/whl/cu121" ;;
#     cu11*) PT_INDEX="https://download.pytorch.org/whl/cu118" ;;
#     *)     PT_INDEX="https://download.pytorch.org/whl/cpu"   ;;
# esac

# # Install PyTorch only if not already present (avoids a large re-download)
# if ! python3 -c "import torch" 2>/dev/null; then
#     info "Installing PyTorch from ${PT_INDEX} …"
#     pip_install torch torchvision torchaudio --index-url "${PT_INDEX}"
# else
#     info "PyTorch already installed: $(python3 -c 'import torch; print(torch.__version__)')"
# fi

# # ── HuggingFace stack ─────────────────────────────────────────────────────────
# info "Installing HuggingFace stack …"
# pip_install \
#     "transformers>=4.36.0" \
#     "datasets>=2.14.0" \
#     "accelerate>=0.25.0" \
#     "sentencepiece>=0.1.99" \
#     "tokenizers>=0.15.0"

# # ── Audio ─────────────────────────────────────────────────────────────────────
# info "Installing audio libraries …"
# pip_install \
#     "librosa>=0.10.0" \
#     "soundfile>=0.12.0" \
#     "audioread>=3.0.0"

# # ── Metrics ───────────────────────────────────────────────────────────────────
# info "Installing evaluation metrics …"
# pip_install "jiwer>=3.0.0"

# # ── Whisper tokenizer extras (best-effort) ────────────────────────────────────
# pip_install "openai-whisper" 2>/dev/null || true

# # ── Optional: NeMo for indic-conformer ───────────────────────────────────────
# # NeMo depends on omegaconf, which is the source of the PyYAML .* warning.
# # We pin omegaconf to a version whose metadata is well-formed, then install NeMo.
# if [[ "${NO_NEMO:-0}" == "1" ]] || [[ "${1:-}" == "--no-nemo" ]]; then
#     warn "--no-nemo / NO_NEMO=1 set; skipping NeMo install."
# else
#     info "Installing omegaconf (pinned) to suppress dependency warning …"
#     pip_install "omegaconf==2.3.0" 2>/dev/null || true

#     info "Attempting NeMo install (for indic-conformer-600m) …"
#     if pip_install "nemo_toolkit[asr]" 2>/dev/null; then
#         info "NeMo installed successfully."
#     else
#         warn "NeMo install failed – indic-conformer will use HF pipeline fallback."
#         warn "To skip NeMo next time: export NO_NEMO=1  or pass --no-nemo"
#     fi
# fi

# # ─── GPU / CUDA status ───────────────────────────────────────────────────────
# $PYTHON - <<'PYEOF'
# import torch
# if torch.cuda.is_available():
#     print(f"[INFO]  CUDA available → {torch.cuda.get_device_name(0)} "
#           f"({torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)")
# else:
#     print("[WARN]  No CUDA detected — running on CPU (will be slow)")
# PYEOF

# # ─── Create output directory ─────────────────────────────────────────────────
# mkdir -p "${OUTPUT_DIR}"

# # ─── Build extra arguments from script args ──────────────────────────────────
# EXTRA_ARGS=""
# for arg in "$@"; do
#     EXTRA_ARGS="${EXTRA_ARGS} ${arg}"
# done

# # ─── Run evaluation ──────────────────────────────────────────────────────────
# info "Launching ASR evaluation …"
# START_TS=$(date +%s)

# $PYTHON "${PYTHON_SCRIPT}" \
#     --audio_dir  "${AUDIO_DIR}"  \
#     --text_file  "${TEXT_FILE}"  \
#     --output_dir "${OUTPUT_DIR}" \
#     ${EXTRA_ARGS}

# END_TS=$(date +%s)
# ELAPSED=$(( END_TS - START_TS ))
# info "Evaluation finished in ${ELAPSED}s"

# # ─── Print final report ───────────────────────────────────────────────────────
# REPORT="${OUTPUT_DIR}/asr_evaluation_report.txt"
# if [[ -f "${REPORT}" ]]; then
#     echo ""
#     echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#     cat "${REPORT}"
#     echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
#     info "Full results saved to: ${OUTPUT_DIR}/"
#     info "Files:"
#     ls -lh "${OUTPUT_DIR}/"
# fi



#!/usr/bin/env bash
# =============================================================================
#  run_asr_evaluation.sh
#  Installs dependencies and runs the full Hindi ASR evaluation pipeline.
#
#  Usage:
#    chmod +x run_asr_evaluation.sh
#    ./run_asr_evaluation.sh                     # run all 4 models
#    ./run_asr_evaluation.sh --models whisper_large_v3  # single model
#    ./run_asr_evaluation.sh --max_utts 50        # quick smoke-test
# =============================================================================

set -euo pipefail

# ─── Paths (edit if needed) ───────────────────────────────────────────────────
AUDIO_DIR="/mnt/storage/aditya/Evaluation/hi_tts_outputs"
TEXT_FILE="/mnt/storage/aditya/Evaluation/Benchmarking_dataset/Hindi_test_set.txt"
OUTPUT_DIR="./asr_results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/run_asr_evaluation.py"

# ─── Colour helpers ──────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ─── Sanity checks ───────────────────────────────────────────────────────────
info "=== Hindi ASR Evaluation Pipeline ==="
info "Audio dir : ${AUDIO_DIR}"
info "Text file : ${TEXT_FILE}"
info "Output    : ${OUTPUT_DIR}"

[[ -d "${AUDIO_DIR}"     ]] || error "Audio directory not found: ${AUDIO_DIR}"
[[ -f "${TEXT_FILE}"     ]] || error "Text file not foundd: ${TEXT_FILE}"
[[ -f "${PYTHON_SCRIPT}" ]] || error "Python script not found: ${PYTHON_SCRIPT}"

NUM_WAV=$(find "${AUDIO_DIR}" -maxdepth 1 \( -name "*.wav" -o -name "*.flac" \) | wc -l)
NUM_REF=$(grep -c . "${TEXT_FILE}" || true)
info "Found ${NUM_WAV} audio files and ${NUM_REF} reference lines."

# ─── Python environment ───────────────────────────────────────────────────────
PYTHON=$(command -v python3 || command -v python || error "python3 not found")
info "Using Python: $($PYTHON --version)"

# Activate conda/venv if present alongside this script
if [[ -f "${SCRIPT_DIR}/venv/bin/activate" ]]; then
    info "Activating local venv …"
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/venv/bin/activate"
elif [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    info "Conda env active: ${CONDA_DEFAULT_ENV}"
fi

# ─── Install Python dependencies ─────────────────────────────────────────────
info "Installing / verifying Python dependencies …"

# Suppress the harmless omegaconf PyYAML .* version-specifier warning that
# appears with older pip resolvers. We capture stderr, filter that one line,
# and re-print everything else so real errors are still visible.
pip_install() {
    local out
    # Run pip; filter only the specific omegaconf PyYAML warning line
    out=$(pip install --no-warn-script-location "$@" 2>&1) || {
        # Non-zero exit: print filtered output then propagate failure
        echo "${out}" | grep -v "Error parsing dependencies of omegaconf" \
                      | grep -v "PyYAML (>=5.1.\*)" >&2
        return 1
    }
    # Zero exit: print filtered output (suppress the benign warning)
    echo "${out}" | grep -v "Error parsing dependencies of omegaconf" \
                  | grep -v "PyYAML (>=5.1.\*)" \
                  | grep -v "^$" \
                  | grep -v "^WARNING: " \
                  || true
}

# Upgrade pip itself quietly
pip install --quiet --upgrade pip 2>/dev/null || true

# ── Detect CUDA version for correct PyTorch wheel ────────────────────────────
# Detection order (most reliable first):
#   0. Already-installed torch that has CUDA → trust it, no reinstall needed
#   1. nvidia-smi        (available even without nvcc on PATH)
#   2. nvcc --version    (when CUDA toolkit bin is on PATH)
#   3. /usr/local/cuda/version.json or version.txt
#   4. /proc/driver/nvidia/version (GPU present, assume CUDA 12.x)
# Manual override: CUDA_TAG=cu121 ./run_asr_evaluation.sh

detect_cuda_tag() {
    # Method 0: torch already installed with CUDA
    local t
    t=$(python3 -c "
import torch
v = torch.version.cuda
if v:
    major, minor = v.split('.')[:2]
    print(f'cu{major}{minor}')
" 2>/dev/null || true)
    [[ -n "$t" ]] && { echo "$t"; return; }

    # Method 1: nvidia-smi
    if command -v nvidia-smi &>/dev/null; then
        local s
        s=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[\d.]+' || true)
        if [[ -n "$s" ]]; then
            echo "cu$(echo "$s" | cut -d. -f1)$(echo "$s" | cut -d. -f2)"; return
        fi
    fi

    # Method 2: nvcc
    if command -v nvcc &>/dev/null; then
        local s
        s=$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' || true)
        if [[ -n "$s" ]]; then
            echo "cu$(echo "$s" | cut -d. -f1)$(echo "$s" | cut -d. -f2)"; return
        fi
    fi

    # Method 3: /usr/local/cuda version files
    local f
    for f in /usr/local/cuda/version.json /usr/local/cuda/version.txt /usr/local/cuda-*/version.txt; do
        [[ -f "$f" ]] || continue
        local s
        s=$(grep -oP '"cuda"\s*:\s*"\K[\d.]+' "$f" 2>/dev/null \
            || grep -oP 'CUDA Version \K[\d.]+' "$f" 2>/dev/null || true)
        if [[ -n "$s" ]]; then
            echo "cu$(echo "$s" | cut -d. -f1)$(echo "$s" | cut -d. -f2)"; return
        fi
    done

    # Method 4: nvidia driver present → assume CUDA 12.1
    [[ -f /proc/driver/nvidia/version ]] && { echo "cu121"; return; }

    echo "cpu"
}

# Allow manual override: CUDA_TAG=cu118 ./run_asr_evaluation.sh
if [[ -n "${CUDA_TAG:-}" ]]; then
    CUDA_VER="${CUDA_TAG}"
    warn "Using manually-set CUDA_TAG=${CUDA_TAG}"
else
    CUDA_VER=$(detect_cuda_tag)
fi

info "Detected compute target: ${CUDA_VER}"

case "${CUDA_VER}" in
    cu12*)  PT_INDEX="https://download.pytorch.org/whl/cu121" ;;
    cu118)  PT_INDEX="https://download.pytorch.org/whl/cu118" ;;
    cu11*)  PT_INDEX="https://download.pytorch.org/whl/cu118" ;;
    *)
        PT_INDEX="https://download.pytorch.org/whl/cpu"
        warn "No GPU detected — inference on 2001 files will be very slow."
        warn "If you have a GPU, override: CUDA_TAG=cu121 ./run_asr_evaluation.sh"
        ;;
esac

# Install / verify PyTorch
TORCH_STATUS=$(python3 -c "
import torch
cuda_ok = torch.cuda.is_available()
print(f'ok cuda_ok={cuda_ok} ver={torch.__version__}')
" 2>/dev/null || echo "missing")

if [[ "${TORCH_STATUS}" == "missing" ]]; then
    info "PyTorch not found — installing from ${PT_INDEX} …"
    pip_install torch torchvision torchaudio --index-url "${PT_INDEX}"
elif echo "${TORCH_STATUS}" | grep -q "cuda_ok=False" && [[ "${CUDA_VER}" != "cpu" ]]; then
    warn "Installed PyTorch is CPU-only but a GPU was detected. Reinstalling with CUDA support …"
    pip_install torch torchvision torchaudio --index-url "${PT_INDEX}"
else
    info "PyTorch already installed: $(python3 -c 'import torch; print(torch.__version__)')"
fi

# ── HuggingFace stack ─────────────────────────────────────────────────────────
info "Installing HuggingFace stack …"
pip_install \
    "transformers>=4.36.0" \
    "datasets>=2.14.0" \
    "accelerate>=0.25.0" \
    "sentencepiece>=0.1.99" \
    "tokenizers>=0.15.0"

# ── Audio ─────────────────────────────────────────────────────────────────────
info "Installing audio libraries …"
pip_install \
    "librosa>=0.10.0" \
    "soundfile>=0.12.0" \
    "audioread>=3.0.0"

# ── Metrics ───────────────────────────────────────────────────────────────────
info "Installing evaluation metrics …"
pip_install "jiwer>=3.0.0"

# ── Whisper tokenizer extras (best-effort) ────────────────────────────────────
pip_install "openai-whisper" 2>/dev/null || true

# ── Optional: NeMo for indic-conformer ───────────────────────────────────────
# NeMo depends on omegaconf, which is the source of the PyYAML .* warning.
# We pin omegaconf to a version whose metadata is well-formed, then install NeMo.
if [[ "${NO_NEMO:-0}" == "1" ]] || [[ "${1:-}" == "--no-nemo" ]]; then
    warn "--no-nemo / NO_NEMO=1 set; skipping NeMo install."
else
    info "Installing omegaconf (pinned) to suppress dependency warning …"
    pip_install "omegaconf==2.3.0" 2>/dev/null || true

    info "Attempting NeMo install (for indic-conformer-600m) …"
    if pip_install "nemo_toolkit[asr]" 2>/dev/null; then
        info "NeMo installed successfully."
    else
        warn "NeMo install failed – indic-conformer will use HF pipeline fallback."
        warn "To skip NeMo next time: export NO_NEMO=1  or pass --no-nemo"
    fi
fi

# ─── GPU / CUDA status ───────────────────────────────────────────────────────
$PYTHON - <<'PYEOF'
import torch
if torch.cuda.is_available():
    print(f"[INFO]  CUDA available → {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)")
else:
    print("[WARN]  No CUDA detected — running on CPU (will be slow)")
PYEOF

# ─── Create output directory ─────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"

# ─── Build extra arguments — strip bash-only flags before passing to Python ───
# Flags consumed ONLY by this shell script; must NOT be forwarded to Python
BASH_ONLY_FLAGS=(--no-nemo)

EXTRA_ARGS=""
for arg in "$@"; do
    skip=0
    for bf in "${BASH_ONLY_FLAGS[@]}"; do
        [[ "${arg}" == "${bf}" ]] && skip=1 && break
    done
    [[ "${skip}" -eq 0 ]] && EXTRA_ARGS="${EXTRA_ARGS} ${arg}"
done

# ─── Run evaluation ──────────────────────────────────────────────────────────
info "Launching ASR evaluation …"
START_TS=$(date +%s)

$PYTHON "${PYTHON_SCRIPT}" \
    --audio_dir  "${AUDIO_DIR}"  \
    --text_file  "${TEXT_FILE}"  \
    --output_dir "${OUTPUT_DIR}" \
    ${EXTRA_ARGS}

END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))
info "Evaluation finished in ${ELAPSED}s"

# ─── Print final report ───────────────────────────────────────────────────────
REPORT="${OUTPUT_DIR}/asr_evaluation_report.txt"
if [[ -f "${REPORT}" ]]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    cat "${REPORT}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    info "Full results saved to: ${OUTPUT_DIR}/"
    info "Files:"
    ls -lh "${OUTPUT_DIR}/"
fi