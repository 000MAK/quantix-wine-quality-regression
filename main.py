"""Compare regression baselines on online UCI Wine Quality data."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Iterable

WINE_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
]


def rmse(actual: Iterable[float], predicted: Iterable[float]) -> float:
    pairs = list(zip(actual, predicted))
    if not pairs:
        raise ValueError("at least one paired observation is required")
    return math.sqrt(sum((float(a) - float(p)) ** 2 for a, p in pairs) / len(pairs))


def mae(actual: Iterable[float], predicted: Iterable[float]) -> float:
    pairs = list(zip(actual, predicted))
    if not pairs:
        raise ValueError("at least one paired observation is required")
    return sum(abs(float(a) - float(p)) for a, p in pairs) / len(pairs)


def main(output_dir: str = "outputs", seed: int = 42) -> dict:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.metrics import r2_score
    import joblib

    frames = []
    for colour, url in zip(("red", "white"), WINE_URLS):
        part = pd.read_csv(url, sep=";")
        part["wine_colour"] = int(colour == "white")
        frames.append(part)
    data = pd.concat(frames, ignore_index=True)
    y = data.pop("quality")
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    models = {
        "median_baseline": DummyRegressor(strategy="median"),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                random_state=seed, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=seed),
    }
    scores, best_name, best_rmse, best_model = {}, None, float("inf"), None
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = rmse(y_test, pred)
        scores[name] = {"mae": mae(y_test, pred), "rmse": score, "r2": float(r2_score(y_test, pred))}
        if score < best_rmse:
            best_name, best_rmse, best_model = name, score, model
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"seed": seed, "test_rows": len(y_test),
        "models": scores, "selected_by_rmse": best_name}, indent=2), encoding="utf-8")
    joblib.dump(best_model, out / "best_model.joblib")
    return scores

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--output-dir", default="outputs"); p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(); print(json.dumps(main(a.output_dir, a.seed), indent=2))
