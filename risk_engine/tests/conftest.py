import os
import sys

RISK_ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRICERS_ROOT = os.path.join(os.path.dirname(RISK_ENGINE_ROOT), "capitolis_pricers", "capitolis_pricers")
for p in (RISK_ENGINE_ROOT, PRICERS_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
