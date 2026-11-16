"""B6 step 1: a drivable road graph for North Carolina from OpenStreetMap, with car travel times.

    ../NC\\ Pipeline/.venv/bin/python b6_roads.py

Input:  data_external/osm/north-carolina-latest.osm.pbf  (Geofabrik extract)
Output: out/b6_graph.npz   directed graph: vertex lat/lon, edges (u, v, seconds)

Speeds follow OSRM's default car profile (car.lua): km/h by highway class, a posted maxspeed when present (mph in the US),
one-way streets respected, private / no-access ways dropped. Turn costs and traffic lights are ignored, so times are a
little optimistic (free-flow). Ways are cut only where they meet another way or end, which keeps the graph small.
"""
import re
import time
from pathlib import Path

import numpy as np
import osmium

HERE = Path(__file__).resolve().parent
PBF = HERE / "data_external/osm/north-carolina-latest.osm.pbf"
OUT = HERE / "out/b6_graph.npz"

SPEED = {"motorway": 90, "motorway_link": 45, "trunk": 85, "trunk_link": 40, "primary": 65, "primary_link": 30,
         "secondary": 55, "secondary_link": 25, "tertiary": 40, "tertiary_link": 20, "unclassified": 25,
         "residential": 25, "living_street": 10, "service": 15}
NO_SERVICE = {"parking_aisle", "drive-through", "emergency_access"}
NO_ACCESS = {"no", "private", "agricultural", "forestry", "delivery"}


def usable(tags) -> bool:
    h = tags.get("highway")
    if h not in SPEED:
        return False
    if tags.get("access") in NO_ACCESS or tags.get("motor_vehicle") in NO_ACCESS or tags.get("motorcar") in NO_ACCESS:
        return False
    if h == "service" and tags.get("service") in NO_SERVICE:
        return False
    if tags.get("area") == "yes":
        return False
    return True


def kmh(tags) -> float:
    base = SPEED[tags.get("highway")]
    m = tags.get("maxspeed")
    if m:
        x = re.match(r"\s*(\d+(?:\.\d+)?)\s*(mph)?", m)
        if x:
            v = float(x.group(1)) * 1.609344          # US limits are posted in mph, with or without the unit
            if 5 <= v <= 130:
                return v
    return base


class Count(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.uses: dict[int, int] = {}

    def way(self, w):
        if not usable(w.tags):
            return
        refs = [n.ref for n in w.nodes]
        if len(refs) < 2:
            return
        u = self.uses
        for r in refs:
            u[r] = u.get(r, 0) + 1
        u[refs[0]] += 1          # endpoints are always vertices
        u[refs[-1]] += 1


class Build(osmium.SimpleHandler):
    def __init__(self, uses):
        super().__init__()
        self.uses = uses
        self.vid: dict[int, int] = {}
        self.lat, self.lon = [], []
        self.eu, self.ev, self.es = [], [], []

    def vertex(self, n):
        i = self.vid.get(n.ref)
        if i is None:
            i = self.vid[n.ref] = len(self.lat)
            self.lat.append(n.location.lat)
            self.lon.append(n.location.lon)
        return i

    def way(self, w):
        t = w.tags
        if not usable(t):
            return
        nodes = list(w.nodes)
        if len(nodes) < 2 or not all(n.location.valid() for n in nodes):
            return
        v = kmh(t) / 3.6                                   # m/s
        ow = t.get("oneway")
        fwd, back = True, True
        if ow in ("yes", "true", "1") or (t.get("highway") in ("motorway",) and ow != "no") or t.get("junction") == "roundabout":
            back = False
        elif ow == "-1":
            fwd, back = False, True
        lat = np.radians([n.location.lat for n in nodes])
        lon = np.radians([n.location.lon for n in nodes])
        seg = 2 * 6371008.8 * np.arcsin(np.sqrt(np.sin(np.diff(lat) / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(np.diff(lon) / 2) ** 2))
        start, acc = self.vertex(nodes[0]), 0.0
        for k in range(1, len(nodes)):
            acc += seg[k - 1]
            if self.uses.get(nodes[k].ref, 0) > 1 or k == len(nodes) - 1:
                end = self.vertex(nodes[k])
                s = acc / v
                if fwd:
                    self.eu.append(start); self.ev.append(end); self.es.append(s)
                if back:
                    self.eu.append(end); self.ev.append(start); self.es.append(s)
                start, acc = end, 0.0


def main():
    t0 = time.time()
    c = Count()
    c.apply_file(str(PBF))
    print(f"pass 1: {len(c.uses):,} way nodes, {time.time() - t0:.0f}s", flush=True)
    b = Build(c.uses)
    b.apply_file(str(PBF), locations=True, idx="flex_mem")
    print(f"pass 2: {len(b.lat):,} vertices, {len(b.eu):,} directed edges, {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(OUT, lat=np.array(b.lat, np.float32), lon=np.array(b.lon, np.float32),
                        u=np.array(b.eu, np.int32), v=np.array(b.ev, np.int32), s=np.array(b.es, np.float32))
    print("saved", OUT, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
