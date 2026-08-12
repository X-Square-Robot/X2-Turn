#!/usr/bin/env bash
set -euo pipefail
PINNED_COMMIT="b1388b1fbf5aaef47937fabe98931211684666a6"
PINNED_VERSION="0.19.1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-}"
if [[ -z "$TARGET" || ! -d "$TARGET/.git" ]]; then
  echo "usage: $0 /path/to/clean/vllm-checkout" >&2; exit 2
fi
actual="$(git -C "$TARGET" rev-parse HEAD)"
if [[ "$actual" != "$PINNED_COMMIT" ]]; then
  echo "refusing: expected vLLM commit $PINNED_COMMIT, got $actual" >&2; exit 1
fi
if [[ -n "$(git -C "$TARGET" status --porcelain)" ]]; then
  echo "refusing: target vLLM checkout is not clean" >&2; exit 1
fi
version_tag="$(git -C "$TARGET" describe --tags --exact-match HEAD 2>/dev/null || true)"
if [[ "$version_tag" != "v$PINNED_VERSION" ]]; then
  echo "refusing: expected vLLM tag v$PINNED_VERSION, got ${version_tag:-none}" >&2
  exit 1
fi
git -C "$TARGET" apply --check "$ROOT/patches/vllm-voxtral-mtp.patch"
git -C "$TARGET" apply "$ROOT/patches/vllm-voxtral-mtp.patch"
install -Dm644 "$ROOT/integrations/vllm/vllm/model_executor/models/voxtral_mtp_utils.py" \
  "$TARGET/vllm/model_executor/models/voxtral_mtp_utils.py"
install -Dm755 "$ROOT/integrations/vllm/tools/export_mtp_for_vllm.py" \
  "$TARGET/tools/export_mtp_for_vllm.py"
mkdir -p "$TARGET/examples/voxtral_mtp"
cp -a "$ROOT/integrations/vllm/examples/voxtral_mtp/." "$TARGET/examples/voxtral_mtp/"
echo "Applied Voxtral MTP overlay to pinned vLLM $PINNED_VERSION ($PINNED_COMMIT)."
