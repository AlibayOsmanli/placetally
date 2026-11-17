"""The 14 object categories and the source labels that map to them."""

INC = {
    "bathtub": {"bathtub"},
    "bed": {"bed", "loft bed"},
    "chair": {
        "chair",
        "armchair",
        "dining chair",
        "desk chair",
        "sofa chair",
        "office chair",
        "kids chair",
        "rocking chair",
        "folding chair",
        "child chair",
        "kids rocking chair",
    },
    "fireplace": {"fireplace"},
    "oven": {"oven"},
    "refrigerator": {"refrigerator", "fridge"},
    "sink": {"sink", "kitchen sink", "washbasin", "basin"},
    "sofa": {"sofa", "couch", "sofa couch", "kitchen sofa"},
    "stool": {"stool", "bar stool", "barstool", "footstool", "kids stool"},
    "stove": {"stove"},
    "table": {
        "table",
        "coffee table",
        "side table",
        "sidetable",
        "couch table",
        "dining table",
        "round table",
        "office table",
    },
    "toilet": {"toilet"},
    "tv_monitor": {"tv", "monitor"},
    "washer": {"washing machine"},
}
CONF = {
    "bed": {"baby bed", "pet bed", "mattress", "bed table"},
    "chair": {
        "stool",
        "bar stool",
        "barstool",
        "footstool",
        "kids stool",
        "bench",
        "corner bench",
        "chairs",
        "folded beach chairs",
        "baby seat",
        "seat",
    },
    "sink": {"sink counter", "bath counter"},
    "sofa": {"armchair", "sofa chair", "bench", "corner bench", "sidecouch"},
    "stool": {"bench", "corner bench", "seat", "baby seat"},
    "table": {
        "desk",
        "computer desk",
        "nightstand",
        "bedside table",
        "dressing table",
        "tv table",
        "tv stand",
        "commode",
        "kitchen counter",
        "counter",
        "stand",
        "foosball table",
        "bed table",
    },
    "tv_monitor": {"pc", "laptop", "screen"},
    "washer": {"clothes dryer", "dishwasher"},
}
CATS = tuple(INC)
LAB2CAT = {}
for _c, _s in INC.items():
    for _l in _s:
        assert _l not in LAB2CAT, _l
        LAB2CAT[_l] = _c
