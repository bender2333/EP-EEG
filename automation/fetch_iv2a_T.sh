#!/usr/bin/env bash
# M13 修改 2：从 GitHub 子仓库只取 IV-2a 的 T 文件（E 文件不下载、不检出），核对 SHA-256，链到 MOABB 的缓存路径。
# 用法：bash automation/fetch_iv2a_T.sh        （仓库根目录下运行；MNE_DATA 默认 /workspace/mne_data）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUB="$ROOT/data/bci-iv-2a"
URL="https://github.com/ParthanOlikkal/BCI_Competition_IV_2a"
PIN="$(git -C "$ROOT" ls-files -s data/bci-iv-2a | awk '{print $2}')"
MNE_DATA="${MNE_DATA:-/workspace/mne_data}"
DEST="$MNE_DATA/MNE-bnci-data/~bci/database/001-2014"

if ! git -C "$SUB" rev-parse HEAD >/dev/null 2>&1; then
  rm -rf "$SUB"   # 空的子仓库目录
  # 部分克隆：只有检出的文件才下载其内容；稀疏检出只要 T 文件，E 文件的内容从不落地
  GIT_LFS_SKIP_SMUDGE=1 git clone -q --filter=blob:none --no-checkout "$URL" "$SUB"
fi
git -C "$SUB" sparse-checkout set --no-cone '/A0[1-9]T.mat' '/README.md'
git -C "$SUB" checkout -q "$PIN"

(cd "$SUB" && sha256sum -c "$ROOT/data/iv2a_T.sha256")
if ls "$SUB" | grep -qiE 'A0[0-9]E'; then echo "E 文件出现在工作区，停止" >&2; exit 1; fi

mkdir -p "$DEST"
for s in 1 2 3 4 5 6 7 8 9; do ln -sf "$SUB/A0${s}T.mat" "$DEST/A0${s}T.mat"; done
echo "T 文件就绪：$DEST"
