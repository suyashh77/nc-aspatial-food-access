"""Step 2: reduce the review embeddings with UMAP (Teresa notebook 2, cells 11 and 14).

    .venv/bin/python step2_umap.py            # GPU (config.ENGINE = "cuml"): a few minutes

Writes
  data/step2_umap5.npy    float32 (n, 5): the space HDBSCAN clusters in (n_neighbors 30, cosine)
  data/step2_umap2.npy    float32 (n, 2): for maps / plots only (n_neighbors 15)
  data/step2_meta.json
"""
from __future__ import annotations

import json
import time

import numpy as np

import config as C


def make_umap(params: dict):
    if C.ENGINE == "cuml":
        from cuml.manifold import UMAP
    else:
        from umap import UMAP
    return UMAP(**params)


def main() -> None:
    emb = np.load(C.DATA / "step1_embeddings.npy", mmap_mode="r")
    meta = {"engine": C.ENGINE, "n": int(emb.shape[0])}
    for name, params in (("umap5", C.UMAP_PARAMS), ("umap2", C.UMAP_VIZ_PARAMS)):
        t0 = time.time()
        out = np.asarray(make_umap(params).fit_transform(np.ascontiguousarray(emb)), dtype=np.float32)
        if not np.isfinite(out).all():
            raise RuntimeError(f"{name}: non-finite coordinates")
        np.save(C.DATA / f"step2_{name}.npy", out)
        meta[name] = {"params": params, "seconds": round(time.time() - t0), "shape": list(out.shape)}
        print(name, meta[name], flush=True)
    (C.DATA / "step2_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
