"""Hand labeler for the gold set: people label individual reviews with the access dimensions they discuss.

This is the yardstick for every decision test (definitions, sentence vs review, mood removal, ...). Today the gold
set is labelled by Claude as a stand-in; this app replaces that with human raters and measures their agreement.

Run:
    cd "/home/scsapre/aspatial_food_access_research/NC Pipeline"
    .venv/bin/streamlit run apps/gold_labeler_app.py --server.port 8502
    # laptop: ssh -L 8502:localhost:8502 <vm>  ->  http://localhost:8502

Input:  decisions/out/gold_sample.csv (gid, id, text, ...). Stars, store and topic are never shown.
Output: decisions/labels/<rater>.csv, one row per review (saved after every review), columns:
        gid, id, rater, no_access_content, <Dim> in {"", "+", "-", "~"} for each dimension, other_subject, confidence,
        notes, seconds, saved_at.
        "+" positive, "-" negative, "~" mixed or neutral mention, "" not discussed.
"""
from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "decisions" / "out" / "gold_sample.csv"
LABELS = ROOT / "decisions" / "labels"
CLAUDE = [ROOT / "decisions" / "out" / f"gold_labels_part{i}.txt" for i in (1, 2)]
DIMS = ["Availability", "Accessibility", "Accommodation", "Affordability", "Acceptability"]
CODE = {"AV": "Availability", "AC": "Accessibility", "AM": "Accommodation", "AF": "Affordability", "AP": "Acceptability"}
CHOICES = {"": "not discussed", "+": "positive", "-": "negative", "~": "mixed / neutral"}
CODEBOOK = {
    "Availability": ("Is the food there? Stock, empty or full shelves, out of stock, selection and variety, the store doesn't "
                     "carry an item, wanting a store (or more stores) in the area.",
                     ["Shelves were empty again.", "Huge selection of produce.", "Wish we had one in our town."]),
    "Accessibility": ("Can people get there and in? Location, distance, close to home, parking, traffic, bus, getting in and out "
                      "of the lot, physical access (wheelchair, mobility carts).",
                      ["Parking lot is a nightmare.", "Close to my house.", "Easy in and out off the highway."]),
    "Accommodation": ("Does the store fit people's lives and needs? Hours and closing early, payment (EBT, SNAP, WIC, cards), "
                      "pickup and delivery, checkout lines and staffing, layout and finding things, dietary options (gluten "
                      "free, halal), store policies.",
                      ["They lock the doors before closing time.", "Pickup took an hour.", "Only two registers open."]),
    "Affordability": ("Prices, value for money, deals, coupons, sales, loyalty discounts, being overcharged.",
                      ["Way overpriced.", "Great deals every week.", "Cheaper than Walmart."]),
    "Acceptability": ("Does it meet people's standards? Freshness and quality of food, cleanliness and smell, how staff treat "
                      "customers (rude, friendly, helpful), safety, cultural fit of the food and the store.",
                      ["The cashier was rude.", "Produce is always fresh.", "Store was filthy."]),
}

st.set_page_config(page_title="Review Labeler", page_icon="🏷️", layout="wide")


@st.cache_data
def load_sample() -> pd.DataFrame:
    s = pd.read_csv(SAMPLE)[["gid", "id", "text"]]
    order = np.random.default_rng(20261110).permutation(len(s))      # same shuffled order for every rater
    return s.iloc[order].reset_index(drop=True)


def rater_file(name: str) -> Path:
    return LABELS / f"{re.sub(r'[^a-z0-9_]+', '_', name.strip().lower())}.csv"


def load_labels(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype={d: str for d in DIMS}).fillna("")
    return pd.DataFrame(columns=["gid", "id", "rater", "no_access_content", *DIMS, "other_subject", "confidence", "notes", "seconds", "saved_at"])


def claude_labels() -> pd.DataFrame:
    rows = []
    for f in CLAUDE:
        if f.exists():
            for line in f.read_text().splitlines():
                gid, *codes = line.split()
                r = {"gid": int(gid), "rater": "claude", "no_access_content": codes == ["none"], **{d: "" for d in DIMS}}
                if codes != ["none"]:
                    for c in codes:
                        r[CODE[c[:2]]] = c[2:]
                rows.append(r)
    return pd.DataFrame(rows)


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    cats = sorted(set(a) | set(b))
    if len(cats) < 2:
        return float("nan")
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


sample = load_sample()
st.sidebar.header("Rater")
rater = st.sidebar.text_input("Your name (labels are saved under it)", key="rater")
st.sidebar.divider()
st.sidebar.header("Codebook")
for d, (txt, ex) in CODEBOOK.items():
    with st.sidebar.expander(d):
        st.markdown(txt)
        st.caption("e.g. " + " · ".join(f'"{e}"' for e in ex))

tab_help, tab_label, tab_agree = st.tabs(["1 · Instructions", "2 · Label reviews", "3 · Agreement"])

with tab_help:
    st.title("Review labeler")
    st.markdown(f"""
You'll read **{len(sample)} Google reviews of North Carolina grocery stores**, one at a time. For each, mark which
of the five dimensions of food access it talks about, and whether it's positive or negative about each.
Your labels become the yardstick for every method choice in this project, so read carefully. Speed isn't the goal.

**For each review**
1. Read the whole review. You won't see the stars or the store, on purpose.
2. For **each dimension**, choose *not discussed*, *positive*, *negative* or *mixed / neutral*.
   - A review can discuss several dimensions, or none.
   - Mark a dimension only if the review actually says something about it. "Great store!" discusses none.
   - "Mixed / neutral" covers a mention that's neither good nor bad ("prices are what you'd expect") or both ("pricey but good sales").
3. If it says nothing about access (pure praise, a gas station, a restaurant visit), tick **No access content**.
4. If it talks about something important that **fits none of the five** (crowding, atmosphere, a pharmacy, Covid rules...),
   write it in **Other subject** in two or three words. This is how we look for dimensions the 5 A's miss.
5. Save. The next review appears. Labels save after every review, so you can stop and come back.

**Judgement calls (follow these so raters agree)**
- Selection, variety, "has everything" → **Availability**.
- Checkout lines, staffing at registers, pickup, delivery, layout, finding items → **Accommodation**.
- Rude, friendly or helpful staff; clean or dirty; fresh or rotten → **Acceptability**.
- Parking and getting to or into the store → **Accessibility**.
- "Close to home" → Accessibility. "Wish there was one near me" → Availability.

About 200–260 reviews takes **2–3 hours**. Two people labelling the same reviews lets us measure agreement (tab 3).
""")

with tab_label:
    if not rater.strip():
        st.info("Enter your name in the sidebar to start.")
        st.stop()
    path = rater_file(rater)
    labels = load_labels(path)
    done = set(labels.gid.astype(int)) if len(labels) else set()
    st.progress(len(done) / len(sample), text=f"{len(done)} of {len(sample)} labelled · saving to {path.name}")
    if "pos" not in st.session_state:
        todo = [i for i, g in enumerate(sample.gid) if g not in done]
        st.session_state.pos = todo[0] if todo else 0
    c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
    if c1.button("⬅️ Previous"):
        st.session_state.pos = max(0, st.session_state.pos - 1)
    if c2.button("Next ➡️"):
        st.session_state.pos = min(len(sample) - 1, st.session_state.pos + 1)
    if c3.button("Next unlabelled"):
        todo = [i for i, g in enumerate(sample.gid) if g not in done]
        if todo:
            st.session_state.pos = next((i for i in todo if i > st.session_state.pos), todo[0])
    c4.markdown(f"Review **{st.session_state.pos + 1}** of {len(sample)}")
    row = sample.iloc[st.session_state.pos]
    gid = int(row.gid)
    prev = labels[labels.gid.astype(int) == gid].iloc[-1].to_dict() if gid in done else {}
    if st.session_state.get("shown_gid") != gid:
        st.session_state.shown_gid, st.session_state.shown_at = gid, time.time()
    st.markdown(f"<div style='font-size:1.15rem; line-height:1.6; padding:1rem; border-radius:8px; "
                f"background:rgba(127,127,127,0.08)'>{row.text}</div>", unsafe_allow_html=True)
    if prev:
        st.caption("✅ already labelled. Saving again overwrites it.")
    with st.form(f"f{gid}"):
        none = st.checkbox("No access content", value=str(prev.get("no_access_content", "")).lower() == "true")
        cols = st.columns(5)
        vals = {}
        for i, d in enumerate(DIMS):
            cur = str(prev.get(d, ""))
            vals[d] = cols[i].radio(d, list(CHOICES), index=list(CHOICES).index(cur) if cur in CHOICES else 0,
                                    format_func=lambda k: CHOICES[k], key=f"{gid}_{d}")
        a, b = st.columns([2, 1])
        other = a.text_input("Other subject (not one of the five)", value=str(prev.get("other_subject", "")))
        conf = b.select_slider("Confidence", ["low", "medium", "high"], value=str(prev.get("confidence", "high")) or "high")
        notes = st.text_input("Notes", value=str(prev.get("notes", "")))
        saved = st.form_submit_button("💾 Save and next", type="primary")
    if saved:
        if none and any(vals.values()):
            st.error("You ticked 'No access content' but also marked a dimension. Untick one.")
        elif not none and not any(vals.values()) and not other.strip():
            st.error("Mark at least one dimension, write an Other subject, or tick 'No access content'.")
        else:
            new = {"gid": gid, "id": row.id, "rater": rater.strip(), "no_access_content": none, **vals, "other_subject": other.strip(),
                   "confidence": conf, "notes": notes, "seconds": round(time.time() - st.session_state.shown_at, 1),
                   "saved_at": datetime.now().isoformat(timespec="seconds")}
            LABELS.mkdir(parents=True, exist_ok=True)
            labels = pd.concat([labels[labels.gid.astype(int) != gid], pd.DataFrame([new])], ignore_index=True)
            labels.to_csv(path, index=False)
            todo = [i for i, g in enumerate(sample.gid) if g not in set(labels.gid.astype(int))]
            st.session_state.pos = next((i for i in todo if i > st.session_state.pos), todo[0] if todo else st.session_state.pos)
            st.rerun()

with tab_agree:
    st.subheader("Agreement between raters")
    files = sorted(LABELS.glob("*.csv")) if LABELS.exists() else []
    raters = {f.stem: load_labels(f) for f in files}
    cl = claude_labels()
    if len(cl):
        raters["claude (stand-in)"] = cl.astype({d: str for d in DIMS})
    names = list(raters)
    if len(names) < 2:
        st.info("Agreement needs at least two raters (Claude's stand-in labels count as one).")
    else:
        a_name = st.selectbox("Rater A", names, index=0)
        b_name = st.selectbox("Rater B", names, index=1)
        A = raters[a_name].assign(gid=lambda x: x.gid.astype(int)).drop_duplicates("gid", keep="last")
        B = raters[b_name].assign(gid=lambda x: x.gid.astype(int)).drop_duplicates("gid", keep="last")
        m = A.merge(B, on="gid", suffixes=("_a", "_b"))
        st.metric("Reviews labelled by both", len(m))
        if len(m) >= 10:
            rows = []
            for d in DIMS:
                ma, mb = (m[f"{d}_a"] != "").to_numpy(), (m[f"{d}_b"] != "").to_numpy()
                both = ma & mb
                rows.append({"dimension": d, "mentioned by A": int(ma.sum()), "mentioned by B": int(mb.sum()),
                             "kappa: mentioned?": round(kappa(ma, mb), 2),
                             "polarity agreement when both mention": round(float(np.mean(m.loc[both, f"{d}_a"] == m.loc[both, f"{d}_b"])), 2) if both.any() else None})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("Cohen's kappa: 1 = perfect, 0 = chance. Above 0.6 is usually called substantial agreement.")
            dis = m[[any((r[f"{d}_a"] != "") != (r[f"{d}_b"] != "") for d in DIMS) for _, r in m.iterrows()]]
            if len(dis):
                st.markdown(f"**{len(dis)} reviews where the raters disagree on a dimension** (discuss these to refine the codebook)")
                txt = sample.set_index("gid").text
                for _, r in dis.head(25).iterrows():
                    fa = ", ".join(f"{d[:6]}{r[f'{d}_a']}" for d in DIMS if r[f"{d}_a"]) or "none"
                    fb = ", ".join(f"{d[:6]}{r[f'{d}_b']}" for d in DIMS if r[f"{d}_b"]) or "none"
                    st.markdown(f"> {txt.get(r.gid, '')[:300]}  \n**A:** {fa} · **B:** {fb}")
    parts = [df[["other_subject"]] for df in raters.values() if "other_subject" in df and len(df)]
    others = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["other_subject"])
    if len(others) and (others.other_subject.astype(str).str.strip() != "").any():
        st.subheader("Other subjects raters wrote in (candidates for dimensions beyond the 5 A's)")
        st.dataframe(others.other_subject.astype(str).str.strip().str.lower().replace("", np.nan).dropna().value_counts().rename("reviews"))
