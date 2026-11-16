"""Build google_drive_upload/, the folder that goes to Google Drive, from the current pipeline outputs.

The split: GitHub holds the code. Drive holds README.docx and docs/, presentation/, labeling/, data/, results/, code/.
data/pipeline_working_files.zip holds the result and working files the code reads, at their paths in the repo,
so a fresh clone plus place_data.sh <data> gives a working project. Docs become Word via handoff/md2docx.py.

    "NC Pipeline/.venv/bin/python" build_drive_package.py            # about 1 minute (+ copying the 0.8 GB database once)

    "NC Pipeline/.venv/bin/python" build_drive_package.py --zips     # also drive_zips/*.zip, for uploading by hand

Re-run it whenever outputs change; run_after_labeling.sh runs it last. It builds a fresh folder, keeps the previous
one as google_drive_upload.prev/, and reuses the archived database when its checksum hasn't changed.
Score files carry "_PROVISIONAL" in their names until the human topic check has been imported.
What goes to Drive and what stays on the server, and why: handoff/FILE_MANIFEST.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
A, B = ROOT / "NC Pipeline", ROOT / "NC Pipeline Improved"
AD, BO = A / "data", B / "out"
HANDOFF = ROOT / "handoff"
DEST = ROOT / "google_drive_upload"
TMP, PREV = ROOT / "google_drive_upload.tmp", ROOT / "google_drive_upload.prev"
ZIPS = ROOT / "drive_zips"
DOCS, DECK, LAB, DATA, RES, CODE = "docs", "presentation", "labeling", "data", "results", "code"
MD = f"{DOCS}/markdown"
PEOPLE_DOCS = ("HANDOVER", "LABELING", "USING_THE_DATA", "DATA_DICTIONARY")
sys.path.insert(0, str(HANDOFF))

# The result and working files that are not in git: folders (or single files) to take, and what to leave out of them
# (large files a script rebuilds, raw rater downloads, dropped Haywood outputs). Keep in step with .gitignore.
WORK_INCLUDE = [
    "NC Pipeline/data", "NC Pipeline/data_external", "NC Pipeline/dash_data",
    "NC Pipeline/decisions/out", "NC Pipeline/decisions/labels",
    "NC Pipeline Improved/out",
    "Teresa Project/Code Pipeline/*.ipynb", "Teresa Project/Code Pipeline/bertopic_model",
    "Teresa Project/Code Pipeline/topic_centroids_frozen.npz",
    "Teresa Project/Code Pipeline/data/definition_text.json",
    "Teresa Project/Code Pipeline/data/manual_validation/manual_validation_scores.csv",
    "Teresa Project/Raleigh_Data/store_scores.csv", "Teresa Project/Raleigh_Data/raleigh_ring_reviews_with_dims.csv",
    "Teresa Project/Raleigh_Data/final_embedding_model", "Teresa Project/Raleigh_Data/definition_embeddings",
    "Teresa Project/Teresa Files/Aspatial_Access_NC_Plan.md", "Teresa Project/Teresa Files/Teresa_Groton_Masters_Thesis.pdf",
    "Reviews Data/*.md",
    "grocery-foot-traffic/NC_Grocery_POIs.xlsx", "grocery-foot-traffic/map/data/nc.geojson",
    "handoff/*.md", "handoff/presentation/Final_Presentation.*",
]
BG_OUTLINES = "grocery-foot-traffic/map/data/bg.geojson"
WORK_EXCLUDE = [
    "NC Pipeline/data/step1_embeddings.npy", "NC Pipeline/data/step1_reviews.parquet", "NC Pipeline/data/step3_grid/*",
    "NC Pipeline/data/exp_no_empty/*", "NC Pipeline/data/web_labels/*", "NC Pipeline/data/*haywood*",
    "NC Pipeline/data/exports/haywood/*", "NC Pipeline/data/exports/NC_reviews_with_dimensions.parquet",
    "NC Pipeline/data_external/census/gaz_tracts_2022.zip",
    "NC Pipeline Improved/out/b2_sentence_embeddings.npy",
    "*/__pycache__/*", "*/.pytest_cache/*",
]
DIMS = ["Availability", "Accessibility", "Accommodation", "Affordability", "Acceptability"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def label_status() -> str:
    meta = json.loads((AD / "step6_meta.json").read_text())
    return "HUMAN" if meta["validity"].startswith("human") else "PROVISIONAL"


def put(src: Path, rel: str) -> None:
    dst = TMP / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def docx(md: Path, rel: str) -> None:
    """Markdown -> styled Word with handoff/md2docx.py; pandoc (plainer) if python-docx isn't installed."""
    dst = TMP / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        from md2docx import convert
    except ImportError:
        subprocess.run(["pandoc", str(md), "-o", str(dst), "--resource-path", str(md.parent)], check=True)
        return
    convert(md, dst)


def doc_pair(md: Path, name: str, word_dir: str = DOCS, md_dir: str = MD) -> None:
    """A doc for people: Word in word_dir, the Markdown source in md_dir."""
    put(md, f"{md_dir}/{name}.md")
    docx(md, f"{word_dir}/{name}.docx")


def main() -> None:
    status = label_status()
    sfx = "_PROVISIONAL" if status == "PROVISIONAL" else ""
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir()

    # 1. Docs: Word for people, Markdown sources in docs/markdown. Technical docs as Markdown only.
    doc_pair(HANDOFF / "README.md", "README", word_dir=".")
    for stem in PEOPLE_DOCS:
        doc_pair(HANDOFF / f"{stem}.md", stem)
    for stem in ("FOR_CLAUDE", "FILE_MANIFEST"):
        put(HANDOFF / f"{stem}.md", f"{MD}/{stem}.md")

    # 2. The presentation
    for f in (HANDOFF / "presentation").glob("Final_Presentation.*"):
        put(f, f"{DECK}/{f.name}")

    # 3. Datasets. The store list
    for f in ("NC_Grocery_POIs_v2.xlsx", "NC_Grocery_POIs_v2.csv", "NC_Grocery_POIs_v1_rows_not_in_v2.csv"):
        put(AD / "exports" / f, f"{DATA}/store_list/{f}")

    # Scores: stores and census tracts, both pipelines
    stores = pd.read_csv(AD / "step6_store_scores.csv")[["store_id", "place_id", "store_name", "city", "county", "category", "lat", "lon"]]
    S = f"{DATA}/scores"
    put(AD / "step6_store_scores.csv", f"{S}/store_scores_A_teresa_method{sfx}.csv")
    put(AD / "step6_store_scores_frozen.csv", f"{S}/store_scores_A_frozen_raleigh_model{sfx}.csv")
    stores.merge(pd.read_csv(BO / "b_store_scores.csv"), on="store_id", how="right").to_csv(
        TMP / S / f"store_scores_B_improved{sfx}.csv", index=False)
    for w in ("", "_equal"):
        local = pd.read_csv(BO / f"b6_store_local_z_b{w}.csv")
        stores.merge(local, on="store_id", how="right").to_csv(TMP / S / f"store_local_z_B{w or '_visits'}{sfx}.csv", index=False)
    put(AD / "step7_tracts_nc.csv", f"{S}/tract_scores_A_1mile{sfx}.csv")
    put(AD / "step7_tracts_frozen.csv", f"{S}/tract_scores_A_frozen_1mile{sfx}.csv")
    put(BO / "b6_tracts_b.csv", f"{S}/tract_scores_B_drive_time_visits{sfx}.csv")
    put(BO / "b6_tracts_b_equal.csv", f"{S}/tract_scores_B_drive_time_equal{sfx}.csv")
    for f in ("step7_regressions_nc.csv", "step7_regressions_frozen.csv"):
        put(AD / f, f"{S}/regressions/{f}")
    for f in ("b6_local_market_a.json", "b6_local_market_a_equal.json", "b6_local_market_b.json", "b6_local_market_b_equal.json",
              "b_pipeline.json"):
        put(BO / f, f"{S}/regressions/{f}")

    # Review level
    for f in ("NC_reviews_with_dimensions.parquet", "NC_reviews_with_dimensions_preview.csv", "NC_reviews_with_dimensions_meta.json"):
        put(AD / "exports" / f, f"{DATA}/review_level/{f.replace('.parquet', sfx + '.parquet')}")

    # Human labeling: the page (to republish), its inputs, and whatever labels have been imported
    L = LAB
    for f in ("labeler.html", "data.js", "data_b2.js", "build_data.py", "build_data_b2.py"):
        put(A / "apps/web_labeler" / f, f"{L}/web_labeler_page/{f}")
    put(A / "decisions/out/gold_sample.csv", f"{L}/inputs/gold_sample_260_reviews.csv")
    put(AD / "step5_validation_topics.csv", f"{L}/inputs/topics_A_83.csv")
    put(AD / "parents_draft.csv", f"{L}/inputs/topics_A_draft_groups.csv")
    put(BO / "b2_validation_topics.csv", f"{L}/inputs/topics_B_sentence_83.csv")
    imported = [*(A / "decisions/labels").glob("*.csv"), *AD.glob("step5_manual_scores*.csv"), *BO.glob("b2_manual_scores*.csv")]
    for f in imported:
        put(f, f"{L}/imported_labels/{f.name}")
    if (AD / "step5_import_meta.json").exists():
        put(AD / "step5_import_meta.json", f"{L}/imported_labels/step5_import_meta.json")
    if not imported:
        (TMP / L / "imported_labels").mkdir(parents=True, exist_ok=True)
        (TMP / L / "imported_labels/NOT_YET.txt").write_text("No human labels have been imported yet. See docs/LABELING.docx.\n")

    # Raleigh case study: the result files here, and a workbook (rebuilt from b_raleigh.json every time) with the reports
    for f in ("b_raleigh.json", "b_raleigh_equal.json", "b_raleigh_stores.csv", "b_raleigh_stores_equal.csv"):
        put(BO / f, f"{DATA}/raleigh_case_study/{f}")
    raleigh_workbook(TMP / RES / f"Raleigh_case_study{sfx}.xlsx")

    # Methods and reference
    doc_pair(ROOT / "PIPELINE_GUIDE.md", "PIPELINE_GUIDE")
    for stem in ("HANDOFF", "STATUS", "README"):
        doc_pair(ROOT / f"Reviews Data/{stem}.md", stem, word_dir=f"{DOCS}/scraping", md_dir=f"{MD}/scraping")
    put(ROOT / "Teresa Project/Teresa Files/Teresa_Groton_Masters_Thesis.pdf", f"{DOCS}/Teresa_Groton_Masters_Thesis.pdf")
    for f in sorted((A / "decisions/out").glob("d*.json")):
        put(f, f"{RES}/decision_tests/{f.name}")
    for f in sorted(BO.glob("b[1-7]*.json")):
        if not f.name.startswith("b6_local_market"):          # already in data/scores/regressions
            put(f, f"{RES}/pipeline_B_tests/{f.name}")

    # The result and working files the code reads (not in git), at their repo paths
    working_files_zip(TMP / DATA / "pipeline_working_files.zip")

    # A small safety copy of the code. The code's home is the GitHub repo.
    (TMP / CODE).mkdir(parents=True, exist_ok=True)
    if (ROOT / ".git").exists():
        subprocess.run(["git", "-C", str(ROOT), "archive", "--format=zip", "--prefix=nc-aspatial-food-access/",
                        "-o", str(TMP / CODE / "nc-aspatial-food-access_code.zip"), "HEAD"], check=True)
    else:
        code_zip(TMP / CODE / "nc-aspatial-food-access_code.zip")

    # The database: the master reviews DB (content = freeze v2) and the side databases
    archive_databases()

    rows = []
    for p in sorted(TMP.rglob("*")):
        if p.is_file():
            rows.append({"path": str(p.relative_to(TMP)), "bytes": p.stat().st_size,
                         "sha256": sha256(p) if p.stat().st_size < 2e9 else ""})
    pd.DataFrame(rows).to_csv(TMP / DATA / "PACKAGE_CONTENTS.csv", index=False)
    (TMP / "LABEL_STATUS.txt").write_text(
        f"Labels: {status}. Built {date.today().isoformat()}.\n"
        + ("Scores use a provisional list of valid topics until the human topic check is imported.\n"
           if status == "PROVISIONAL" else "Scores use the human topic check.\n"))
    if PREV.exists():
        shutil.rmtree(PREV)
    if DEST.exists():
        DEST.rename(PREV)
    TMP.rename(DEST)
    total = sum(r["bytes"] for r in rows)
    print(f"google_drive_upload/: {len(rows)} files, {total / 1e9:.2f} GB, labels {status}")
    if "--zips" in sys.argv:
        upload_zips()


def working_files() -> list[Path]:
    """Every file under WORK_INCLUDE that WORK_EXCLUDE doesn't rule out."""
    from fnmatch import fnmatch
    out = set()
    for pat in WORK_INCLUDE:
        for p in ROOT.glob(pat):
            for f in ([p] if p.is_file() else p.rglob("*")):
                rel = f.relative_to(ROOT).as_posix()
                if f.is_file() and not any(fnmatch(rel, x) for x in WORK_EXCLUDE):
                    out.add(f)
    return sorted(out)


def working_files_zip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for f in working_files():
            z.write(f, f.relative_to(ROOT))
        # Block-group outlines, which stage 6 uses to find each store's county. The server's file also carries
        # foot-traffic summaries (licensed source), so only the public part goes in: outline, block-group id, county.
        bg = json.loads((ROOT / BG_OUTLINES).read_text())
        for feat in bg["features"]:
            feat["properties"] = {k: feat["properties"][k] for k in ("id", "county")}
        z.writestr(BG_OUTLINES, json.dumps(bg, separators=(",", ":")))


def upload_zips() -> None:
    """Two zips for uploading by hand: docs.zip (everything but data/) and data.zip. Each unzips into
    NC_Aspatial_Food_Access/, so unzipping both in one place gives the single folder to drag into Drive."""
    if ZIPS.exists():
        shutil.rmtree(ZIPS)
    ZIPS.mkdir()
    top = "NC_Aspatial_Food_Access"
    groups = {"docs": [DOCS, DECK, LAB, RES, CODE], "data": [DATA]}
    for name, folders in groups.items():
        with zipfile.ZipFile(ZIPS / f"{name}.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            if name == "docs":                                 # the loose files at the top travel with the documents
                for f in sorted(DEST.iterdir()):
                    if f.is_file():
                        z.write(f, f"{top}/{f.name}")
            for folder in folders:
                for f in sorted((DEST / folder).rglob("*")):
                    if f.is_file():
                        z.write(f, f"{top}/{f.relative_to(DEST)}")
        print(f"  drive_zips/{name}.zip  {(ZIPS / f'{name}.zip').stat().st_size / 1e6:.0f} MB")


def raleigh_workbook(path: Path) -> None:
    rows, agree, r8 = [], [], []
    for w, f in (("visits", "b_raleigh.json"), ("equal", "b_raleigh_equal.json")):
        d = json.loads((BO / f).read_text())
        for scope in ("within_1mi_of_her_stores", "all_wake"):
            t = d["tracts"][scope]
            for dim in DIMS:
                x = t[dim]
                rows.append({"store_weights": w, "tracts": scope, "n_tracts": t["n_tracts"], "dimension": dim,
                             "teresa_b_pct_black": x["teresa_b_black"], "A_b_pct_black": x["A_b_black"], "A_p": x["A_p"],
                             "B_b_pct_black": x["B_b_black"], "B_p": x["B_p"]})
        for dim in DIMS:
            x = d["tracts_8min"][dim]
            r8.append({"store_weights": w, "dimension": dim, "A_b_pct_black": x["A"]["b_black"], "A_p": x["A"]["p"],
                       "B_b_pct_black": x["B"]["b_black"], "B_p": x["B"]["p"], "n_tracts": x["A"]["n"]})
        if w == "visits":
            for dim in DIMS:
                x = d["store_agreement_with_teresa"][dim]
                agree.append({"dimension": dim, "stores": x["stores"], "rho_A_frozen": x["A_frozen"], "rho_A_nc_refit": x["A_nc"],
                              "rho_B": x["B"], "top3_teresa": "; ".join(x["top3_teresa"]), "top3_B": "; ".join(x["top3_B"])})
    with pd.ExcelWriter(path) as xw:
        pd.DataFrame({"read_me": [
            "Raleigh case study. Rank agreement (rho) is Spearman, against Teresa's own pooled store scores for her 42 matched stores.",
            "Tract tables: population-weighted regression of each tract score on % Black (as a share) and income ($10k).",
            "b = change in the standardized tract score for a tract that is 100% vs 0% Black. Divide by 10 for a 10-point difference.",
            "8-minute markets: each tract's typical store within an 8-minute drive. These are the version to present for Raleigh.",
            "store_weights 'visits' uses Advan foot traffic; 'equal' counts every reachable store the same.",
            f"Labels: {label_status()}."]}).to_excel(xw, sheet_name="read_me", index=False)
        pd.DataFrame(agree).to_excel(xw, sheet_name="store_agreement", index=False)
        pd.DataFrame(r8).to_excel(xw, sheet_name="tracts_8min_markets", index=False)
        pd.DataFrame(rows).to_excel(xw, sheet_name="tracts_1mile_like_teresa", index=False)
        pd.read_csv(BO / "b_raleigh_stores.csv").to_excel(xw, sheet_name="store_scores_42", index=False)


def code_zip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keep_ext = {".py", ".md", ".html", ".js", ".sh", ".txt", ".json", ".toml"}
    skip_dirs = {".venv", "__pycache__", ".pytest_cache", "data", "out", "data_external", "logs", "dash_data"}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for base in (A, B):
            for p in base.rglob("*"):
                rel = p.relative_to(ROOT)
                if p.is_file() and p.suffix in keep_ext and not (set(rel.parts) & skip_dirs):
                    z.write(p, rel)
        for f in ("run_after_labeling.sh", "build_drive_package.py", "place_data.sh", "PIPELINE_GUIDE.md", "CLAUDE.md", "README.md"):
            z.write(ROOT / f, f)
        z.write(HANDOFF / "FOR_CLAUDE.md", "handoff/FOR_CLAUDE.md")
        z.write(ROOT / "requirements_lab_vm.txt", "requirements_lab_vm.txt")


def archive_databases() -> None:
    D = f"{DATA}/database"
    (TMP / D).mkdir(parents=True)
    rd = ROOT / "Reviews Data"
    files = ["Step_5_master_reviews.db", "Step_5_out_of_scope.db", "Step_5_closed_stores.db", "Step_5_excluded_stores.db",
             "Step_1_NC_Food_Retail_Frame.db"]
    sums = []
    for f in files:
        src, old = rd / f, PREV / D / f if not DEST.exists() else DEST / D / f
        h = sha256(src)
        if old.exists() and sha256(old) == h:
            os.link(old, TMP / D / f)           # unchanged: reuse the earlier copy instead of copying 0.7 GB again
        else:
            shutil.copy2(src, TMP / D / f)
        sums.append(f"{h}  {f}")
    big = AD / "step1_reviews.parquet"                        # cleaned review text, a pipeline input (also a release asset)
    shutil.copy2(big, TMP / D / big.name)
    sums.append(f"{sha256(big)}  {big.name}")
    (TMP / D / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")


if __name__ == "__main__":
    sys.exit(main())
