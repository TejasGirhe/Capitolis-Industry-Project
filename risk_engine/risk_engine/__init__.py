"""
risk_engine -- Monte Carlo simulation engine built on top of capitolis_pricers.

Sibling package: depends on capitolis_pricers as a library, never modifies it.
"""
from .factors.types import RateFactor, EquityFactor, FxFactor, CreditFactor, FactorSet
from .factors.extract import extract_factors
from .models.registry import get_model, MODEL_REGISTRY, get_rate_model, get_spot_model, RATE_MODEL_REGISTRY, SPOT_MODEL_REGISTRY
from .simulate import simulate
from .simulation.joint import JointSimulator, JointSimResult
from .simulation.scenario_market import build_market_states_at
from .simulation.grid import build_simulation_grid, SimulationGrid, reporting_anchors, collect_regression_dates, trade_cashflow_dates
from .simulation.interpolate import interpolate_state, bridge_weight
from .simulation.per_trade import (
    extract_trade_factors, build_trade_simulator, simulate_trade, simulate_all_trades_independently,
)
from .pricing import price_curves, price_curves_per_trade, CurveResult
from .netting import (
    NettingSet, MarginModel, ZeroMargin, Counterparty, build_netting_hierarchy,
    IndependentAggregateProfile, aggregate_independent_profiles, aggregate_all_independent_profiles,
)
from .exposure import (
    ExposureProfile, compute_exposure_profile, compute_all_profiles,
    compute_per_trade_exposure_profile, compute_per_trade_profiles,
)
from .plotting import plot_exposure_profiles
from .visualization import plot_factor_fan, plot_all_factor_fans, plot_counterparty_cashflow_graph, plot_all_cashflow_graphs
from .xva import XVAResult, compute_cva, compute_dva, compute_fva, compute_xva_report, compute_all_xva

__all__ = [
    "RateFactor", "EquityFactor", "FxFactor", "CreditFactor", "FactorSet",
    "extract_factors", "get_model", "MODEL_REGISTRY",
    "get_rate_model", "get_spot_model", "RATE_MODEL_REGISTRY", "SPOT_MODEL_REGISTRY",
    "simulate", "JointSimulator", "JointSimResult", "build_market_states_at",
    "build_simulation_grid", "SimulationGrid", "reporting_anchors", "collect_regression_dates", "trade_cashflow_dates",
    "interpolate_state", "bridge_weight",
    "extract_trade_factors", "build_trade_simulator", "simulate_trade", "simulate_all_trades_independently",
    "price_curves", "price_curves_per_trade", "CurveResult",
    "NettingSet", "MarginModel", "ZeroMargin", "Counterparty", "build_netting_hierarchy",
    "IndependentAggregateProfile", "aggregate_independent_profiles", "aggregate_all_independent_profiles",
    "ExposureProfile", "compute_exposure_profile", "compute_all_profiles",
    "compute_per_trade_exposure_profile", "compute_per_trade_profiles",
    "plot_exposure_profiles",
    "plot_factor_fan", "plot_all_factor_fans", "plot_counterparty_cashflow_graph", "plot_all_cashflow_graphs",
    "XVAResult", "compute_cva", "compute_dva", "compute_fva", "compute_xva_report", "compute_all_xva",
]
