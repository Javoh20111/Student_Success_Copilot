from .dataset import generate_dataset, load_dataset

def train_models(df):
    from .model import train_models as _tm
    return _tm(df)

def evaluate_models(models, X_test, y_test):
    from .model import evaluate_models as _em
    return _em(models, X_test, y_test)

def predict_risk(profile, models):
    from .model import predict_risk as _pr
    return _pr(profile, models)

def prepare_data(df, **kw):
    from .model import prepare_data as _pd
    return _pd(df, **kw)

def run_full_ml_pipeline(df=None):
    from .model import run_full_ml_pipeline as _rp
    return _rp(df)

__all__ = [
    "generate_dataset", "load_dataset",
    "train_models", "evaluate_models",
    "predict_risk", "prepare_data",
    "run_full_ml_pipeline",
]