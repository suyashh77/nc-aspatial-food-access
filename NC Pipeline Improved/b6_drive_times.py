"""B6 step 2: drive time from every NC census tract (the "person") to every store within 32 minutes.

    ../NC\\ Pipeline/.venv/bin/python b6_drive_times.py

Origin of a person = the tract's internal point (Census Gazetteer 2022), as in stage 7. Each origin and each store is
snapped to the nearest road vertex in the largest connected part of the graph; the snap distance is added at 20 km/h
(driveway / parking lot). Dijkstra from every tract with a 32-minute cutoff.

Output: out/b6_drive_times.parquet   GEOID, store_id, minutes   (pairs within 32 min)
        out/b6_drive_meta.json
"""
import json
import sqlite3
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from common_b import C, OUT

CUTOFF_MIN = 32.0
SNAP_KMH = 20.0


def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)] * 6371008.8


def main():
    t0 = time.time()
    g = np.load(OUT / "b6_graph.npz")
    n = len(g["lat"])
    G = sp.csr_matrix((g["s"].astype(np.float64) / 60.0, (g["u"], g["v"])), shape=(n, n))   # minutes
    G.sum_duplicates()
    _, comp = connected_components(G, directed=True, connection="strong")
    big = np.bincount(comp).argmax()
    keep = np.flatnonzero(comp == big)
    print(f"graph {n:,} vertices; largest strongly connected part {len(keep):,}", flush=True)
    tree = cKDTree(xyz(g["lat"][keep], g["lon"][keep]))

    tr = pd.read_csv(C.CENSUS / "2022_Gaz_tracts_national.txt", sep="\t", dtype={"GEOID": str})
    tr.columns = [c.strip() for c in tr.columns]
    tr = tr[tr.GEOID.str.startswith("37")][["GEOID", "INTPTLAT", "INTPTLONG"]].rename(columns={"INTPTLAT": "lat", "INTPTLONG": "lon"})
    con = sqlite3.connect(f"file:{C.REVIEWS_DB}?mode=ro", uri=True)
    st = pd.read_sql("SELECT store_id, source_lat AS lat, source_lon AS lon FROM stores", con).dropna()

    dt, it = tree.query(xyz(tr.lat.to_numpy(), tr.lon.to_numpy()))
    ds, is_ = tree.query(xyz(st.lat.to_numpy(), st.lon.to_numpy()))
    tv, sv = keep[it], keep[is_]
    t_snap = dt / 1000 / SNAP_KMH * 60
    s_snap = ds / 1000 / SNAP_KMH * 60
    # stores by vertex (several stores can share one)
    by_v = pd.DataFrame({"v": sv, "store_id": st.store_id.to_numpy(), "snap": s_snap})
    uv = np.unique(sv)
    col = {v: i for i, v in enumerate(uv)}

    rows = []
    B = 64
    for s0 in range(0, len(tv), B):
        src = tv[s0:s0 + B]
        D = dijkstra(G, directed=True, indices=src, limit=CUTOFF_MIN)[:, uv]          # minutes, inf beyond cutoff
        D = D + t_snap[s0:s0 + B, None]
        for k in range(len(src)):
            ok = np.isfinite(D[k])
            if not ok.any():
                continue
            m = pd.DataFrame({"v": uv[ok], "m": D[k, ok]}).merge(by_v, on="v")
            m["minutes"] = m.m + m.snap
            m = m[m.minutes <= CUTOFF_MIN]
            rows.append(pd.DataFrame({"GEOID": tr.GEOID.iat[s0 + k], "store_id": m.store_id.to_numpy(), "minutes": m.minutes.astype(np.float32).to_numpy()}))
        if s0 % (B * 10) == 0:
            print(f"{s0 + len(src):,}/{len(tv):,} tracts, {time.time() - t0:.0f}s", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(OUT / "b6_drive_times.parquet", index=False)
    per = out.groupby("GEOID").size()
    meta = {"tracts": int(len(tr)), "stores": int(len(st)), "pairs_within_32": int(len(out)),
            "tracts_with_a_store_16min": int(out[out.minutes <= 16].GEOID.nunique()),
            "tracts_with_a_store_32min": int(out.GEOID.nunique()),
            "median_stores_within_16": float(out[out.minutes <= 16].groupby("GEOID").size().reindex(tr.GEOID, fill_value=0).median()),
            "median_tract_snap_m": float(np.median(dt)), "median_store_snap_m": float(np.median(ds)),
            "seconds": round(time.time() - t0)}
    (OUT / "b6_drive_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
