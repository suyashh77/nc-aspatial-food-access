"""Inline SVG maps of North Carolina for the deck. Read from the current data files, so they update after labeling.

Colors: categorical slots 1-3 of the reference palette (blue, orange, aqua; validated: CVD dE 9.2, normal 27.6),
diverging blue <-> red with a gray midpoint. Outline: Census county and state lines (grocery-foot-traffic/map/data/nc.geojson).
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NC = ROOT / "grocery-foot-traffic/map/data/nc.geojson"
POIS = ROOT / "NC Pipeline/data/exports/NC_Grocery_POIs_v2.csv"
TRACTS_B = ROOT / "NC Pipeline Improved/out/b6_tracts_b.csv"
LON0, LAT0, LON1, LAT1 = -84.35, 33.75, -75.40, 36.62
OUT_OF_STATE = (", GA ", ", SC ", ", VA ", ", TN ")
COS = math.cos(math.radians(35.5))
BLUE, ORANGE, AQUA, RED, MID = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#e3e1dc"


def _proj(w):
    k = w / ((LON1 - LON0) * COS)
    return (lambda lon, lat: ((lon - LON0) * COS * k, (LAT1 - lat) * k)), round((LAT1 - LAT0) * k)


def _outline(p):
    d = json.loads(NC.read_text())
    county, state = [], []
    for f in d["features"]:
        kind, g = f["properties"].get("kind"), f["geometry"]
        if kind not in ("county", "state"):
            continue
        lines = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]
        for ln in lines:
            pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (p(*c[:2]) for c in ln))
            (county if kind == "county" else state).append(f'<polyline points="{pts}"/>')
    return (f'<g fill="none" stroke="#d5d7da" stroke-width="0.8">{"".join(county)}</g>',
            f'<g fill="none" stroke="#8a8f96" stroke-width="1.4">{"".join(state)}</g>')


def _svg(w, h, county, body, state):
    return (f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" role="img">'
            f'{county}{body}{state}</svg>')


def store_groups():
    rows = list(csv.DictReader(POIS.open()))
    def grp(c):
        if c == "Dollar store":
            return "dollar"
        if c in ("Grocery store", "Supermarket"):
            return "grocery"
        return "other"
    return rows, grp


def stores_map(w=1180):
    p, h = _proj(w)
    county, state = _outline(p)
    rows, grp = store_groups()
    color = {"grocery": BLUE, "dollar": ORANGE, "other": AQUA}
    dots, counts = {"other": [], "grocery": [], "dollar": []}, {"grocery": 0, "dollar": 0, "other": 0, "left_off": 0}
    for r in rows:
        if not r["latitude"] or any(s in r["full_address"] for s in OUT_OF_STATE):
            counts["left_off"] += 1
            continue
        x, y = p(float(r["longitude"]), float(r["latitude"]))
        g = grp(r["category"])
        counts[g] += 1
        dots[g].append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6"/>')
    body = "".join(f'<g fill="{color[g]}" fill-opacity="0.8" stroke="#fff" stroke-width="0.5">{"".join(v)}</g>' for g, v in dots.items())
    return _svg(w, h, county, body, state), counts


def _tracts():
    return list(csv.DictReader(TRACTS_B.open()))


def reach_map(w=1180):
    p, h = _proj(w)
    county, state = _outline(p)
    have, none16, none32 = [], [], []
    people16 = people32 = 0
    for t in _tracts():
        x, y = p(float(t["lon"]), float(t["lat"]))
        if int(float(t["n_stores_16"] or 0)) == 0:
            people16 += int(float(t["pop"] or 0))
            ring = int(float(t["n_stores_32"] or 0)) == 0
            if ring:
                people32 += int(float(t["pop"] or 0))
            (none32 if ring else none16).append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6"/>')
        else:
            have.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.4"/>')
    body = (f'<g fill="#c9cdd2">{"".join(have)}</g>'
            f'<g fill="{ORANGE}" stroke="#fff" stroke-width="1">{"".join(none16)}</g>'
            f'<g fill="{ORANGE}" stroke="#111" stroke-width="2">{"".join(none32)}</g>')
    return _svg(w, h, county, body, state), {"tracts_none16": len(none16) + len(none32), "tracts_none32": len(none32),
                                              "people16": people16, "people32": people32}


def _diverge(v, lim=2.0):
    """Blue for higher-than-typical scores, red for lower, gray in the middle."""
    t = max(-1.0, min(1.0, v / lim))
    a, b = (BLUE, MID) if t >= 0 else (RED, MID)
    t = abs(t)
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(cb[i] + (ca[i] - cb[i]) * t):02x}" for i in range(3))


def score_map(dim="Acceptability", w=1180):
    p, h = _proj(w)
    county, state = _outline(p)
    dots, n = [], 0
    rows = sorted(_tracts(), key=lambda t: abs(float(t[f"E_{dim}_16"] or 0)))   # strongest colors drawn last
    for t in rows:
        v = t[f"E_{dim}_16"]
        if not v:
            continue
        n += 1
        x, y = p(float(t["lon"]), float(t["lat"]))
        dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="{_diverge(float(v))}"/>')
    body = f'<g stroke="#fff" stroke-width="0.4">{"".join(dots)}</g>'
    return _svg(w, h, county, body, state), n


def diverge_legend():
    stops = [-2, -1, 0, 1, 2]
    cells = "".join(f'<span style="flex:1;height:16px;background:{_diverge(s)}"></span>' for s in stops)
    return (f'<div style="width:420px"><div style="display:flex;gap:2px">{cells}</div>'
            '<div style="display:flex;justify-content:space-between;font-size:18px;color:#5b6067;margin-top:6px">'
            '<span>2 SD lower</span><span>typical</span><span>2 SD higher</span></div></div>')
