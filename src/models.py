import numpy as np
import joblib

from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# MODEL CREATION
# ============================================================

def create_models():
    """
    Create all regression models used by AlphaTrade.
    """

    models = {

        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror"
        )
    }

    return models


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(model, X_train, y_train):
    """
    Train a single model.
    """

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# PREDICT
# ============================================================

def predict_model(model, X_test):
    """
    Generate predictions.
    """

    predictions = model.predict(
        X_test
    )

    return predictions


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):
    """
    Evaluate regression model using
    MAE, RMSE and R2.
    """

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# TRAIN ALL MODELS
# ============================================================

def train_all_models(
    X_train,
    y_train,
    X_test,
    y_test
):
    """
    Train and evaluate all models.
    """

    models = create_models()

    results = {}
    trained_models = {}

    for name, model in models.items():

        print(
            f"\nTraining {name}..."
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        results[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }

        trained_models[name] = model

        print(
            f"{name} trained successfully"
        )

        print(
            f"MAE  : {mae:.4f}"
        )

        print(
            f"RMSE : {rmse:.4f}"
        )

        print(
            f"R2   : {r2:.4f}"
        )

    return trained_models, results


# ============================================================
# FIND BEST MODEL
# ============================================================

def get_best_model(
    trained_models,
    results
):
    """
    Select model with lowest RMSE.
    """

    best_model_name = min(
        results,
        key=lambda name: results[name]["RMSE"]
    )

    best_model = trained_models[
        best_model_name
    ]

    return (
        best_model_name,
        best_model
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    filename="best_model.pkl"
):
    """
    Save trained model inside models/.
    """

    model_dir = Path("models")

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        model_dir / filename
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"Model saved to: {model_path}"
    )

    return model_path


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    filename="best_model.pkl"
):
    """
    Load saved model.
    """

    model_path = (
        Path("models")
        / filename
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = joblib.load(
        model_path
    )

    print(
        f"Model loaded from: {model_path}"
    )

    return model


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(
    model,
    feature_names
):
    """
    Return feature importance for tree models.
    """

    if not hasattr(
        model,
        "feature_importances_"
    ):

        return None

    importance = model.feature_importances_

    result = {
        "Feature": feature_names,
        "Importance": importance
    }

    return result