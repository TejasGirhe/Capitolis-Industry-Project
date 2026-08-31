"""
Name -> model class registries, split by factor kind (RateModel vs SpotModel)
to match the type-safe base-class split in models/base.py -- a caller asking
for a rate model by name can never accidentally get a spot model back.
"""
from .lgm import LGM1F, LGM1FSV, LGM2F, LGM2FSV
from .gbm import GBM, GBM_SV
from .fx import FXGBM, FXGBM_SV

RATE_MODEL_REGISTRY = {
    "LGM1F": LGM1F,
    "LGM1F_SV": LGM1FSV,
    "LGM2F": LGM2F,
    "LGM2F_SV": LGM2FSV,
}

SPOT_MODEL_REGISTRY = {
    "GBM": GBM,
    "GBM_SV": GBM_SV,
    "FXGBM": FXGBM,
    "FXGBM_SV": FXGBM_SV,
}


def get_rate_model(name: str, **kwargs):
    """Instantiate a registered rate model by name, e.g. get_rate_model('LGM2F_SV', mean_reversion_a1=0.05)."""
    key = name.strip().upper()
    if key not in RATE_MODEL_REGISTRY:
        raise KeyError(f"Unknown rate model '{name}'. Available: {sorted(RATE_MODEL_REGISTRY)}")
    return RATE_MODEL_REGISTRY[key](**kwargs)


def get_spot_model(name: str, **kwargs):
    """Instantiate a registered spot model by name, e.g. get_spot_model('GBM_SV')."""
    key = name.strip().upper()
    if key not in SPOT_MODEL_REGISTRY:
        raise KeyError(f"Unknown spot model '{name}'. Available: {sorted(SPOT_MODEL_REGISTRY)}")
    return SPOT_MODEL_REGISTRY[key](**kwargs)


# Backward-compat aliases for the prior (rate-only) single-registry API used
# by risk_engine/simulate.py and the earlier test suite.
MODEL_REGISTRY = RATE_MODEL_REGISTRY
get_model = get_rate_model
