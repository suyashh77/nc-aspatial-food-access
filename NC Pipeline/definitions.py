"""The two versions of Teresa's dimension definitions, plus test variants.

third_person: Code Pipeline/data/definition_text.json (= the saved definition_embeddings.npy).
first_person: Groton_Thesis.pptx slide 71 ("Embedded Definitions").
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIMS = ["Availability", "Accessibility", "Accommodation", "Affordability", "Acceptability"]

THIRD_PERSON = json.loads((ROOT / "Teresa Project/Code Pipeline/data/definition_text.json").read_text())

FIRST_PERSON = {
    "Availability": "There is a grocery store in my neighborhood, and it has plenty of food for the people who live nearby. "
    "The store is big enough that all the people who live here can easily get food here. The store has all the food that I need. "
    "The shelves are well stocked and there are usually not empty shelves. When I need basic items, this store has them available.",
    "Accessibility": "I can physically get to this store easily. It is close to my house, and it is not difficult to drive there. "
    "I can also take the bus, bike, or walk to my destination. The parking lot is really convenient and I never have a hard time "
    "getting to this grocery store. The layout of this store is really organized.",
    "Accommodation": "This store is open at the hours I need, but sometimes it closes early. It has a lot of payment options and "
    "takes apple pay, SNAP, WIC, and other payment options without any issues. There are gluten free and sugar free options to "
    "accommodate my needs.",
    "Affordability": "The prices at this store are great. Everything in the store is worth the cost. Some things are overpriced "
    "but I can use my savings card and get good deals on products. They have sales and discounts that make the prices more affordable.",
    "Acceptability": "People judge this store by its friendly employees, fresh, high quality food, and clean bathrooms. Rude customer "
    "service and dirty stores are unacceptable. The food here is always fresh and high quality, especially the produce. There is a "
    "great selection and variety of my favorite foods.",
}


def sentences(d: dict) -> dict:
    """Split each definition into its sentences (for the 'average of short sentences' variant)."""
    import re
    return {k: [s.strip() for s in re.split(r"(?<=[.!?])\s+", " ".join(v.split())) if len(s.strip()) > 3] for k, v in d.items()}

# Thesis slide 9: the short academic definitions from the literature (Caspi 2012; Turner 2021).
LITERATURE = {
    "Availability": "Sufficient supply of grocery options in the physical area, and food is stocked in the store.",
    "Accessibility": "Ease of reaching grocery store in terms of location, encompasses distance, transportation options, and travel time.",
    "Accommodation": "How well grocery options accept and adapt to residents' needs.",
    "Affordability": "Food prices and people's perception of worth relative to the cost.",
    "Acceptability": "Whether grocery store meets personal standards of residents.",
}

VERSIONS = {"first_person": FIRST_PERSON, "third_person": THIRD_PERSON, "literature": LITERATURE}


def embed_definitions(model, version: str = "first_person", mode: str = "paragraph"):
    """5 x 384 unit vectors. mode='paragraph' embeds each definition whole (Teresa);
    mode='sentences' embeds each sentence and averages them (a test variant)."""
    import numpy as np
    d = VERSIONS[version]
    if mode == "paragraph":
        return model.encode([d[k] for k in DIMS], normalize_embeddings=True)
    out = []
    for k, ss in sentences(d).items():
        v = model.encode(ss, normalize_embeddings=True).mean(0)
        out.append(v / np.linalg.norm(v))
    return np.stack(out)
