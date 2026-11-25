#!/usr/bin/env bash
# Frees about 42 GB on the server, as Prof. Mandl asked. Nothing here is read by any script behind a
# thesis figure (checked 20 and 25 Nov 2026). Deletion cannot be undone, so the script only lists
# what it would remove unless it is run with --yes.
#
#   bash scripts/clean_server_disk.sh          # list only
#   bash scripts/clean_server_disk.sh --yes    # delete
#
# Removed:
#   archive/cleanup_2026_01_05/                      ~24 GB  January 2026 pipeline outputs, not in git, unreferenced
#   CycleGAN per-epoch checkpoints (N_net_*.pth)     ~18 GB  the latest_net_* weights of each run are kept
# Kept on purpose: self_annotation_package.tar.gz (a backup of the hand labels, which cannot be regenerated).
set -u
cd "$(dirname "$0")/.."
CK=pytorch-CycleGAN-and-pix2pix/checkpoints
targets=()
[ -d archive/cleanup_2026_01_05 ] && targets+=(archive/cleanup_2026_01_05)
while IFS= read -r f; do targets+=("$f"); done < <(find "$CK" -maxdepth 2 -name '[0-9]*_net_*.pth' 2>/dev/null | sort)
if [ ${#targets[@]} -eq 0 ]; then echo "nothing to remove"; exit 0; fi
du -shc "${targets[@]}" 2>/dev/null | tail -1
echo "${#targets[@]} paths (archive folder + per-epoch checkpoints)"
if [ "${1:-}" = "--yes" ]; then
  rm -rf -- "${targets[@]}" && echo "removed"
  ls "$CK"/*/latest_net_* 2>/dev/null | sed 's/^/kept: /'
else
  echo "list only; run with --yes to delete"
fi
