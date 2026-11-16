"""Human topic check for the NC topic model (Pipeline A, step 5; Teresa's notebook 4 app, extended with a merge step).

Run:
    cd "/home/scsapre/aspatial_food_access_research/NC Pipeline"
    .venv/bin/python step5_validation_input.py          # once, builds data/step5_validation_topics.csv
    .venv/bin/streamlit run apps/topic_validation_app.py --server.port 8501

Then open http://localhost:8501 (on the VM, forward the port: ssh -L 8501:localhost:8501 <vm>).

Writes (data/):
    step5_merge_decisions.csv   one row per topic: group, status (valid / invalid / needs_review), reason
    step5_group_scores.csv      one row per group: status, 0-3 score per dimension, notes (blinded)
    step5_manual_scores.csv     one row per topic in Teresa's column format (Topic, manual_topic_status,
                                manual_invalid_reason, manual_<Dim>..., manual_notes, last_updated):
                                each topic inherits its group's scores. Step 6 reads this file.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parents[1] / "data"
INPUT = DATA / "step5_validation_topics.csv"
MERGE = DATA / "step5_merge_decisions.csv"
GROUPS = DATA / "step5_group_scores.csv"
TOPIC_SCORES = DATA / "step5_manual_scores.csv"

DIMS = ["Availability", "Accessibility", "Accommodation", "Affordability", "Acceptability"]
RUBRIC = {0: "not represented", 1: "weakly or ambiguously represented", 2: "clearly represented", 3: "strongly represented"}
STATUS = ["valid", "invalid", "needs_review"]
REASONS = ["", "too_generic", "too_short_or_low_information", "not_food_access_related", "mixed_or_incoherent",
           "duplicate_or_redundant", "other"]
GENERIC_PREFIXES = ("General praise", "No clear subject")
DIM_GUIDE = {
    "Availability": "Is there enough food, and is it on the shelves? Stock, empty shelves, selection and variety, items the store doesn't carry.",
    "Accessibility": "Can people get there? Distance, location, parking, bus, traffic, getting in and out.",
    "Accommodation": "Does the store fit people's lives? Hours, payment (SNAP, WIC, EBT), pickup and delivery, checkout and lines, layout, dietary options.",
    "Affordability": "Prices, deals, coupons, value for money.",
    "Acceptability": "Does it meet people's standards? Freshness and quality, cleanliness, how staff treat customers, cultural fit.",
}

st.set_page_config(page_title="NC Topic Check", page_icon="🗂️", layout="wide")


# ------------------------------------------------------------------ data
@st.cache_data
def load_topics() -> pd.DataFrame:
    if not INPUT.exists():
        st.error(f"Missing {INPUT}. Run `.venv/bin/python step5_validation_input.py` first.")
        st.stop()
    return pd.read_csv(INPUT).fillna("").sort_values("n_reviews", ascending=False).reset_index(drop=True)


def load_merge(topics: pd.DataFrame) -> pd.DataFrame:
    if MERGE.exists():
        m = pd.read_csv(MERGE).fillna("")
        missing = set(topics.Topic) - set(m.Topic)
        if not missing:
            return m
    # first run: pre-fill from the draft groups; generic groups start as invalid / too_generic
    m = topics[["Topic", "draft_parent"]].rename(columns={"draft_parent": "group"})
    m["group"] = m["group"].replace("", "Ungrouped")
    generic = m["group"].str.startswith(GENERIC_PREFIXES)
    m["status"] = ["invalid" if g else "valid" for g in generic]
    m["reason"] = ["too_generic" if g else "" for g in generic]
    m["note"] = ""
    m["confirmed"] = False
    return m


def save_merge(m: pd.DataFrame) -> None:
    m.to_csv(MERGE, index=False)


def load_groups() -> pd.DataFrame:
    if GROUPS.exists():
        return pd.read_csv(GROUPS).fillna("")
    return pd.DataFrame(columns=["group", "status", "reason", *[f"manual_{d}" for d in DIMS], "notes", "last_updated"])


def export_topic_scores(m: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """Teresa's per-topic format. Invalid topics (merge step) stay invalid with zeros; valid topics take their
    group's scores; topics whose group is not scored yet are left out (step 6 treats them as unscored)."""
    rows = []
    gs = g.set_index("group") if len(g) else pd.DataFrame()
    for _, r in m.iterrows():
        if r.status == "invalid":
            rows.append({"Topic": r.Topic, "manual_topic_status": "invalid", "manual_invalid_reason": r.reason,
                         **{f"manual_{d}": 0 for d in DIMS}, "manual_notes": r.note, "group": r.group,
                         "last_updated": datetime.now().isoformat(timespec="seconds")})
        elif len(gs) and r.group in gs.index:
            s = gs.loc[r.group]
            rows.append({"Topic": r.Topic, "manual_topic_status": s.status, "manual_invalid_reason": s.reason,
                         **{f"manual_{d}": int(s[f"manual_{d}"]) for d in DIMS}, "manual_notes": s.notes,
                         "group": r.group, "last_updated": s.last_updated})
    out = pd.DataFrame(rows)
    if len(out):
        out.sort_values("Topic").to_csv(TOPIC_SCORES, index=False)
    return out


def review(label: str, text: str) -> None:
    if str(text).strip():
        st.markdown(f"**{label}**")
        st.info(str(text).strip())


topics = load_topics()
tix = topics.set_index("Topic")
if "merge" not in st.session_state:
    st.session_state.merge = load_merge(topics)
merge: pd.DataFrame = st.session_state.merge
groups = load_groups()

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Scoring rubric")
for k, v in RUBRIC.items():
    st.sidebar.markdown(f"**{k}** = {v}")
st.sidebar.divider()
st.sidebar.header("The five dimensions")
for d, txt in DIM_GUIDE.items():
    st.sidebar.markdown(f"**{d}**: {txt}")
st.sidebar.divider()
st.sidebar.caption(f"Files: {MERGE.name}, {GROUPS.name}, {TOPIC_SCORES.name} in {DATA}")

tab_help, tab_merge, tab_score, tab_done = st.tabs(["1 · Instructions", "2 · Merge topics", "3 · Score groups", "4 · Progress & export"])

# ------------------------------------------------------------------ 1 instructions
with tab_help:
    st.title("NC topic check")
    st.markdown(f"""
This is the human check from Teresa's method (her notebook 4), for the **{len(topics)} NC topics**.
She marked 18 of her 59 Raleigh topics invalid. Your scores decide which topics count toward store scores,
and they are the yardstick for the model's topic-to-dimension matching. **The model's guesses are hidden on purpose.**

**Step 2 · Merge topics** (about 30 min)
1. Topics start in the draft groups from `parents_draft.csv`. Near-duplicates (e.g. five "clean + staff" topics) sit together.
2. Open each group. Read the keywords and the sample reviews of every topic in it.
3. If a topic doesn't belong, move it to another group or type a new group name.
4. **Groups named "General praise" or "No clear subject" start as invalid / too_generic.** Keep them invalid unless the
   sample reviews show a real subject. Topic 0, for example, has a generic label but real content in its samples. If so, set it to valid
   and put it in a group.
5. Tick **Confirmed** on each group when you're done. Merging only decides what you score together; it doesn't change the topic model.

**Step 3 · Score groups** (about 2 min per group)
1. For each valid group, choose **valid**, **invalid** (with a reason) or **needs_review**.
2. Score **each dimension 0–3 on its own**. These are not ranks; a group can score 3 on two dimensions.
3. Judge what the reviews are *about*, not whether they're positive. "Prices are crazy" and "great deals" both score 3 on Affordability.
4. Use the sidebar definitions. When in doubt, write a note.
5. Save. The app moves to the next group.

**Step 4 · Progress & export** writes `step5_manual_scores.csv` in Teresa's format. Step 6 (store scores) reads it.
For agreement statistics, a second person can score the same groups into a copy of the files (change the file names at the top of the app).
""")

# ------------------------------------------------------------------ 2 merge
with tab_merge:
    st.subheader("Merge near-duplicate topics into groups")
    all_groups = sorted(merge.group.unique())
    c1, c2, c3 = st.columns(3)
    c1.metric("Topics", len(merge))
    c2.metric("Groups", merge[merge.status != "invalid"].group.nunique())
    c3.metric("Topics invalid", int((merge.status == "invalid").sum()))
    show = st.radio("Show", ["all groups", "not confirmed", "invalid"], horizontal=True)
    order = merge.groupby("group").Topic.apply(lambda s: tix.loc[s, "n_reviews"].sum()).sort_values(ascending=False).index
    for grp in order:
        sub = merge[merge.group == grp]
        if show == "not confirmed" and sub.confirmed.all():
            continue
        if show == "invalid" and not (sub.status == "invalid").any():
            continue
        n = int(tix.loc[sub.Topic, "n_reviews"].sum())
        flag = "✅" if sub.confirmed.all() else "⬜"
        with st.expander(f"{flag} {grp} · {len(sub)} topics · {n:,} reviews"):
            for _, r in sub.iterrows():
                t = tix.loc[r.Topic]
                st.markdown(f"**Topic {r.Topic}** · {t.n_reviews:,} reviews · ★ {t.mean_stars} · _{t.Representation}_")
                if t.draft_note:
                    st.caption(f"Draft note: {t.draft_note}")
                cols = st.columns(2)
                with cols[0]:
                    for i in (1, 2, 3):
                        if t[f"top_review_{i}"]:
                            st.markdown(f"> {t[f'top_review_{i}'][:400]}")
                with cols[1]:
                    for i in (1, 2, 3):
                        if t[f"random_review_{i}"]:
                            st.markdown(f"> {t[f'random_review_{i}'][:400]}")
                k1, k2, k3, k4 = st.columns([2, 2, 1, 2])
                choice = k1.selectbox("Group", all_groups + ["➕ new group…"], index=all_groups.index(r.group), key=f"g{r.Topic}")
                if choice == "➕ new group…":
                    choice = k1.text_input("New group name", key=f"ng{r.Topic}") or r.group
                status = k2.selectbox("Status", STATUS, index=STATUS.index(r.status), key=f"s{r.Topic}")
                reason = k4.selectbox("Reason (if invalid)", REASONS, index=REASONS.index(r.reason) if r.reason in REASONS else 0, key=f"r{r.Topic}")
                i = merge.index[merge.Topic == r.Topic][0]
                merge.loc[i, ["group", "status", "reason"]] = [choice, status, reason if status == "invalid" else ""]
                st.divider()
            if st.button(f"Confirm group: {grp}", key=f"c_{grp}"):
                bad = merge[(merge.group == grp) & (merge.status == "invalid") & (merge.reason == "")]
                if len(bad):
                    st.error(f"Topics {list(bad.Topic)} are invalid without a reason.")
                else:
                    merge.loc[merge.group == grp, "confirmed"] = True
                    save_merge(merge)
                    st.success("Saved.")
                    st.rerun()
    if st.button("💾 Save all merge decisions", type="primary"):
        save_merge(merge)
        st.success(f"Saved to {MERGE}")

# ------------------------------------------------------------------ 3 score
with tab_score:
    valid_groups = (merge[merge.status != "invalid"].groupby("group").Topic.apply(lambda s: tix.loc[s, "n_reviews"].sum())
                    .sort_values(ascending=False).index.tolist())
    if not valid_groups:
        st.info("No valid groups yet.")
        st.stop()
    done = set(groups.group) if len(groups) else set()
    st.progress(len(done & set(valid_groups)) / len(valid_groups), text=f"{len(done & set(valid_groups))} of {len(valid_groups)} groups scored")
    if "gi" not in st.session_state:
        st.session_state.gi = 0
    st.session_state.gi = min(st.session_state.gi, len(valid_groups) - 1)
    a, b, c = st.columns([1, 3, 1])
    if a.button("⬅️ Previous"):
        st.session_state.gi = max(0, st.session_state.gi - 1)
    if c.button("Next ➡️"):
        st.session_state.gi = min(len(valid_groups) - 1, st.session_state.gi + 1)
    pick = b.selectbox("Group", valid_groups, index=st.session_state.gi)
    st.session_state.gi = valid_groups.index(pick)
    grp = pick
    members = merge[(merge.group == grp) & (merge.status != "invalid")].Topic.tolist()
    tt = tix.loc[members]
    left, right = st.columns([1, 2])
    with left:
        st.subheader(grp + (" ✅" if grp in done else ""))
        st.metric("Reviews", f"{int(tt.n_reviews.sum()):,}")
        st.metric("Mean stars", f"{(tt.mean_stars * tt.n_reviews).sum() / tt.n_reviews.sum():.2f}")
        for t, r in tt.iterrows():
            st.markdown(f"**Topic {t}**: {r.Representation}")
    with right:
        st.subheader("Example reviews")
        for t, r in tt.iterrows():
            with st.expander(f"Topic {t}", expanded=len(tt) <= 2):
                for i in (1, 2, 3):
                    review(f"Most typical {i}", r[f"top_review_{i}"])
                for i in (1, 2, 3):
                    review(f"Random {i}", r[f"random_review_{i}"])
    saved = groups.set_index("group").loc[grp].to_dict() if grp in done else {}
    with st.form(f"form_{grp}"):
        status = st.radio("Group status", STATUS, index=STATUS.index(saved.get("status", "valid")), horizontal=True)
        reason = st.selectbox("Reason (if invalid)", REASONS, index=REASONS.index(saved.get("reason", "")) if saved.get("reason", "") in REASONS else 0)
        st.markdown("**How strongly is each dimension represented?** Score each one on its own.")
        cols = st.columns(5)
        scores = {d: cols[i].radio(d, [0, 1, 2, 3], index=int(saved.get(f"manual_{d}", 0) or 0),
                                   format_func=lambda x: f"{x} · {RUBRIC[x]}", key=f"{grp}_{d}") for i, d in enumerate(DIMS)}
        notes = st.text_area("Notes", value=str(saved.get("notes", "")))
        ok = st.form_submit_button("💾 Save group", type="primary")
    if ok:
        if status == "invalid" and not reason:
            st.error("Pick a reason for an invalid group.")
        else:
            row = {"group": grp, "status": status, "reason": reason if status == "invalid" else "",
                   **{f"manual_{d}": int(v) for d, v in scores.items()}, "notes": notes,
                   "last_updated": datetime.now().isoformat(timespec="seconds")}
            groups = pd.concat([groups[groups.group != grp], pd.DataFrame([row])], ignore_index=True)
            groups.to_csv(GROUPS, index=False)
            save_merge(merge)
            export_topic_scores(merge, groups)
            st.session_state.gi = min(len(valid_groups) - 1, st.session_state.gi + 1)
            st.rerun()

# ------------------------------------------------------------------ 4 progress
with tab_done:
    st.subheader("Progress")
    valid_groups = merge[merge.status != "invalid"].group.unique()
    scored = set(groups.group) if len(groups) else set()
    c1, c2, c3 = st.columns(3)
    c1.metric("Groups confirmed (merge)", f"{merge.groupby('group').confirmed.all().sum()} / {merge.group.nunique()}")
    c2.metric("Groups scored", f"{len(scored & set(valid_groups))} / {len(valid_groups)}")
    c3.metric("Topics invalid", int((merge.status == "invalid").sum()) + int(groups[groups.status == 'invalid'].shape[0]) if len(groups) else int((merge.status == "invalid").sum()))
    if st.button("Write step5_manual_scores.csv", type="primary"):
        out = export_topic_scores(merge, groups)
        st.success(f"Wrote {len(out)} topics to {TOPIC_SCORES}")
    if len(groups):
        st.dataframe(groups, use_container_width=True)
