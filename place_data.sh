#!/usr/bin/env bash
# Puts the data from Google Drive into a fresh clone of this repo, where the code expects it.
#
#   ./place_data.sh <path to the data folder downloaded from Drive (called 3_Datasets in copies before Nov 16, 2026)>
#
# It unzips pipeline_working_files.zip over the repo (results, working files, Teresa's files, the handoff documents),
# then copies in the five databases, the cleaned reviews and the review-level export. Needs about 2.5 GB of disk.
# The databases are checked against database/SHA256SUMS.txt first.
#
# Don't run it in a folder that already has pipeline outputs (the lab VM's project folder, say): it would overwrite
# them with the Drive copy. It refuses unless you add --force.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
FORCE=0 SRC=""
for a in "$@"; do
  case "$a" in
    --force) FORCE=1 ;;
    *) SRC="$a" ;;
  esac
done
[[ -n "$SRC" ]] || { sed -n 2,11p "$0"; exit 2; }
SRC="$(realpath "$SRC")"
for need in pipeline_working_files.zip database/SHA256SUMS.txt review_level; do
  [[ -e "$SRC/$need" ]] || { echo "Not the Drive data folder: $SRC/$need is missing."; exit 2; }
done
if [[ -e "$ROOT/NC Pipeline/data/step2_umap5.npy" && $FORCE -eq 0 ]]; then
  echo "This folder already has pipeline outputs. Running this would overwrite them with the Drive copy."
  echo "Add --force if that is what you want."; exit 1
fi

echo "Checking the databases..."
(cd "$SRC/database" && sha256sum -c SHA256SUMS.txt)

echo "Unzipping the working files..."
unzip -q -o "$SRC/pipeline_working_files.zip" -d "$ROOT"

echo "Copying the large files:"
mkdir -p "$ROOT/Reviews Data" "$ROOT/NC Pipeline/data/exports"
for db in "$SRC"/database/*.db; do
  cp -f "$db" "$ROOT/Reviews Data/"; echo "  Reviews Data/$(basename "$db")"
done
cp -f "$SRC/database/step1_reviews.parquet" "$ROOT/NC Pipeline/data/"; echo "  NC Pipeline/data/step1_reviews.parquet"
# The export is named ..._PROVISIONAL.parquet until the hand labels are in; the code wants the plain name.
export_file="$(ls "$SRC"/review_level/NC_reviews_with_dimensions*.parquet | head -1)"
cp -f "$export_file" "$ROOT/NC Pipeline/data/exports/NC_reviews_with_dimensions.parquet"
echo "  NC Pipeline/data/exports/NC_reviews_with_dimensions.parquet"
echo "Done. Next: read handoff/FOR_CLAUDE.md (code) or handoff/HANDOVER.md, LABELING.md, USING_THE_DATA.md (people)."
