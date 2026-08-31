"""
Thin calibration wrapper: model name + curve + vol surface -> CalibratedModel.

Kept separate from models.registry so calibration inputs (curve, vol_surface,
priors) are assembled in one obvious place, and so simulate.py's entry point
stays a one-liner.
"""
def calibrate(model_name: str, curve, vol_surface, model_kwargs=None, **calibrate_kwargs):
    """calibrate('LGM2F_SV', curve, vol_surface) -> CalibratedModel.

    model_kwargs: passed to the model's constructor (e.g. mean_reversion_a).
    calibrate_kwargs: passed to model.calibrate() (e.g. an explicit sv_prior override).

    Imports models.registry lazily: models.lgm imports calibration.priors, so a
    module-level import here would create calibration <-> models circular import.
    """
    from ..models.registry import get_model
    model = get_model(model_name, **(model_kwargs or {}))
    return model.calibrate(curve, vol_surface, **calibrate_kwargs)
