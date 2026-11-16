#!/usr/bin/env bash
# Downloads the large files (too big for git) from the repo's "handoff" release and puts them where the code expects.
#
#   ./fetch_data.sh            # needs the GitHub CLI, logged in with access to the repo: gh auth login
#
# About 1 GB. Each file is checked against SHA256SUMS.txt.
set -euo pipefail
REPO="suyashh77/nc-aspatial-food-access"
TAG="handoff-2026-11-15"
ROOT="$(cd "$(dirname "$0")" && pwd)"
TMP="$ROOT/.release_download"
rm -rf "$TMP" && mkdir -p "$TMP"       # start empty, so no overwrite flag is needed (older gh versions lack one)
gh release download "$TAG" -R "$REPO" -D "$TMP"
(cd "$TMP" && sha256sum -c SHA256SUMS.txt)

place() { mkdir -p "$(dirname "$ROOT/$2")"; mv -f "$TMP/$1" "$ROOT/$2"; echo "  $2"; }
echo "Placing files:"
for db in Step_5_master_reviews.db Step_5_out_of_scope.db Step_5_closed_stores.db Step_5_excluded_stores.db Step_1_NC_Food_Retail_Frame.db; do
  place "$db" "Reviews Data/$db"
done
place step1_reviews.parquet "NC Pipeline/data/step1_reviews.parquet"
place NC_reviews_with_dimensions.parquet "NC Pipeline/data/exports/NC_reviews_with_dimensions.parquet"
rm -rf "$TMP"
echo "Done. Next: read handoff/README.md."
