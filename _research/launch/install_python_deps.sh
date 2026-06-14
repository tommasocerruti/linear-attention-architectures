#!/bin/bash
set -euo pipefail
PKG_DIR=${1:?usage: install_python_deps.sh PKG_DIR}
MARKER=$PKG_DIR/.deps_installed_v13
LOCK=$PKG_DIR/.install.lockdir

mkdir -p "$PKG_DIR"

clean_stale_numpy_metadata() {
    shopt -s nullglob
    local metadata
    for metadata in "$PKG_DIR"/numpy-*.dist-info/METADATA; do
        if ! grep -q '^Version:' "$metadata"; then
            rm -rf "$(dirname "$metadata")"
        fi
    done
    local dist_info
    for dist_info in "$PKG_DIR"/numpy-*.dist-info; do
        if [ ! -f "$dist_info/METADATA" ]; then
            rm -rf "$dist_info"
        fi
    done
    shopt -u nullglob
}

have_required_deps() {
    PYTHONPATH="$PKG_DIR${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY' >/dev/null 2>&1
import datasets
import sentencepiece
import transformers
import wandb
from fla.modules.convolution import causal_conv1d
from fla.modules.l2norm import l2norm
from fla.ops.kda import chunk_kda
from fla.ops.delta_rule import chunk_delta_rule
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
import tilelang
PY
}

if [ -f "$MARKER" ] && have_required_deps; then exit 0; fi

while ! mkdir "$LOCK" 2>/dev/null; do
    sleep 5
    if [ -f "$MARKER" ] && have_required_deps; then exit 0; fi
done
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

clean_stale_numpy_metadata

if [ -f "$MARKER" ] && have_required_deps; then exit 0; fi

if command -v uv >/dev/null 2>&1; then
    INSTALL="uv pip install --quiet"
else
    INSTALL="pip install --quiet"
fi

# Install tilelang first (with deps — it needs tvm/tvm_ffi), then overwrite
# any packages it clobbers (datasets, numpy) with our pinned versions after.
$INSTALL --upgrade --target="$PKG_DIR" tilelang
$INSTALL --upgrade --target="$PKG_DIR" transformers wandb datasets sentencepiece einops
clean_stale_numpy_metadata
$INSTALL --upgrade --no-deps --target="$PKG_DIR" \
    'causal-conv1d~=1.5' \
    'fla-core==0.5.0' \
    'flash-linear-attention==0.5.0'
$INSTALL --upgrade --no-deps --target="$PKG_DIR" \
    'git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.2.0'

if ! have_required_deps; then
    echo "Dependency validation failed after install. Re-running imports to show the first error:" >&2
    PYTHONPATH="$PKG_DIR${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
import datasets
import sentencepiece
import transformers
import wandb
from fla.modules.convolution import causal_conv1d
from fla.modules.l2norm import l2norm
from fla.ops.kda import chunk_kda
from fla.ops.delta_rule import chunk_delta_rule
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
import tilelang
PY
fi

touch "$MARKER"
