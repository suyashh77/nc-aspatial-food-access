#!/usr/bin/env bash
# Everything that has to be re-run once the human labels are in, in order. About 1.5 hours on the lab VM (GPU).
#
#   ./run_after_labeling.sh --csv  <folder>     # the Results-tab downloads (no Claude needed), or
#   ./run_after_labeling.sh --json <folder>     # the page's database pulled by Claude (gold/, topics/, topics_b2/ inside)
#   ./run_after_labeling.sh --skip-import       # labels already imported; just re-run the analysis
#   add --with-b5 to also re-run the B5 pooling test (adds about 1 hour)
#
# Each step's output goes to logs/after_labeling_<time>/<step>.log. The script stops at the first failure.
# Details: handoff/FOR_CLAUDE.md (section "After labeling") and PIPELINE_GUIDE.md.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
A="$ROOT/NC Pipeline"
B="$ROOT/NC Pipeline Improved"
PY="$A/.venv/bin/python"
MODE="" SRC="" WITH_B5=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --csv|--json) MODE="${1#--}"; SRC="$(realpath "$2")"; shift 2 ;;
    --skip-import) MODE="skip"; shift ;;
    --with-b5) WITH_B5=1; shift ;;
    *) echo "unknown option: $1"; sed -n 2,9p "$0"; exit 2 ;;
  esac
done
[[ -n "$MODE" ]] || { sed -n 2,9p "$0"; exit 2; }

LOG="$ROOT/logs/after_labeling_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG"
step() {  # step <name> <dir> <command...>
  local name="$1" dir="$2"; shift 2
  printf '%s  %-34s' "$(date +%H:%M:%S)" "$name"
  local t0=$SECONDS
  if (cd "$dir" && "$@") >"$LOG/$name.log" 2>&1; then
    echo "ok ($((SECONDS - t0)) s)"
  else
    echo "FAILED, see $LOG/$name.log"; tail -20 "$LOG/$name.log"; exit 1
  fi
}

# 1. Import the labels -> rater files, gold consensus, topic consensus (stage 6 and Pipeline B read these)
case "$MODE" in
  csv)  step 01_import_labels "$A" "$PY" step5_import_web_labels.py --csv-dir "$SRC" ;;
  json) step 01_import_labels "$A" "$PY" step5_import_web_labels.py --src "$SRC" ;;
esac
if [[ ! -f "$A/data/step5_manual_scores.csv" ]]; then
  echo "No data/step5_manual_scores.csv: the topic check isn't imported, so stage 6 would still use the provisional list."
  echo "Import the topic check first (see handoff/LABELING.md)."; exit 1
fi

# 2. Pipeline A: store scores (both arms, statewide and Raleigh), then tract scores
step 02_stage6_nc              "$A" "$PY" step6_store_scores.py
step 03_stage6_frozen          "$A" "$PY" step6_store_scores.py --arm frozen
step 04_stage6_raleigh         "$A" "$PY" step6_store_scores.py --stores raleigh --tag raleigh
step 05_stage6_raleigh_frozen  "$A" "$PY" step6_store_scores.py --arm frozen --stores raleigh --tag raleigh_frozen
step 06_stage7_nc              "$A" "$PY" step7_tracts.py
step 07_stage7_frozen          "$A" "$PY" step7_tracts.py --arm frozen

# 3. Decision tests (switch to the human gold set on their own once 200+ reviews have a consensus)
step 08_d1_definitions         "$A" "$PY" decisions/d1_definitions.py
step 09_d2_d7                  "$A" "$PY" decisions/d2_d7.py all
step 10_d8_raleigh             "$A" "$PY" decisions/d8_raleigh_reproduction.py
step 11_d9_table53             "$A" "$PY" decisions/d9_table53.py
step 12_d10_umap               "$A" "$PY" decisions/d10_definitions_in_umap.py

# 4. Pipeline B: definitions re-tuned on the human gold set, then scores, drive-time markets, Raleigh
step 13_b4_definitions         "$B" "$PY" b4_definitions.py
if [[ $WITH_B5 == 1 ]]; then step 14_b5_pooling "$B" "$PY" b5_pooling.py; fi
step 15_b_pipeline             "$B" "$PY" b_pipeline.py
step 16_b6_scores_a            "$B" "$PY" b6_local_market.py --scores a
step 17_b6_scores_a_equal      "$B" "$PY" b6_local_market.py --scores a --weights equal
step 18_b6_scores_b            "$B" "$PY" b6_local_market.py --scores b
step 19_b6_scores_b_equal      "$B" "$PY" b6_local_market.py --scores b --weights equal
step 20_b_raleigh              "$B" "$PY" b_raleigh.py
step 21_b_raleigh_equal        "$B" "$PY" b_raleigh.py --weights equal

# 5. The review-level export and the Google Drive folder
step 22_export_reviews         "$A" "$PY" export_review_table.py
step 23_drive_package          "$ROOT" "$PY" build_drive_package.py --zips

echo
echo "Done. Logs: $LOG"
echo "google_drive_upload/ now holds the final, human-labeled files. Upload it to Google Drive in place of the old folder"
echo "(drive_zips/ has the same content as one zip per folder, easier to download from the VM)."
echo "Update PIPELINE_GUIDE.md with the new numbers (the json files in NC Pipeline/decisions/out and NC Pipeline Improved/out)."
