"""Two-stage play outcome model for the RL environment.

Usage:
    model = PlayOutcomeModel('data/')
    outcome, yards = model.predict(
        quarter=2, down=3, yardsToGo=7, gameClock_seconds=420,
        absoluteYardlineNumber=65, isDropback=True, passRushers=4,
        score_diff=-3,
        offenseFormation='SHOTGUN', playType='pass',
        defFormation='Nickel (4-2-5)', pff_passCoverage='Cover-3',
        pff_manZone='Zone', designedPass='medium_middle',
        receiverAlignment='3x1', pff_runConceptPrimary='none',
    )
"""

import json
import pathlib
import random

import joblib
import numpy as np


class PlayOutcomeModel:
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Auto-detect: look for model files in same directory as this module
            data_path = pathlib.Path(__file__).parent
        else:
            data_path = pathlib.Path(data_dir)

        # Load metadata
        with open(data_path / "model_features.json") as f:
            self._feat_data = json.load(f)
        with open(data_path / "label_encoders.json") as f:
            self._enc_data = json.load(f)

        self._feature_cols = self._feat_data["feature_cols"]
        self._numeric_features = self._feat_data["numeric_features"]
        self._categorical_features = self._feat_data["categorical_features"]
        self._cat_indices = self._feat_data["categorical_indices"]
        self._outcome_mapping = self._feat_data["outcome_mapping"]
        self._quantile_levels = self._feat_data["quantile_levels"]

        # Build category-to-code lookup for each categorical feature
        self._cat_encoders = {}
        for col in self._categorical_features:
            categories = self._enc_data[col]
            self._cat_encoders[col] = {v: i for i, v in enumerate(categories)}

        # Load classifier
        self._clf = joblib.load(data_path / "outcome_classifier.joblib")

        # Load quantile regressors
        self._regressors = {}
        for oc, reg_type in self._feat_data["outcome_regressor_types"].items():
            if reg_type == "quantile":
                models = {}
                for q in self._quantile_levels:
                    models[q] = joblib.load(data_path / f"yards_q{q:.2f}_{oc}.joblib")
                self._regressors[oc] = {"type": "quantile", "models": models}
            else:
                with open(data_path / f"yards_{oc}.json") as f:
                    emp = json.load(f)
                self._regressors[oc] = {"type": "empirical", "yards": emp["yards"]}

    def _encode_features(self, **kwargs) -> np.ndarray:
        """Encode raw feature values into the model's expected format."""
        # Derive binary features if not provided
        down = kwargs.get("down", 1)
        yards_to_go = kwargs.get("yardsToGo", 10)
        field_pos = kwargs.get("absoluteYardlineNumber", 25)

        if "is_third_down_long" not in kwargs:
            kwargs["is_third_down_long"] = int(down == 3 and yards_to_go >= 7)
        if "red_zone" not in kwargs:
            kwargs["red_zone"] = int(field_pos >= 80)

        row = []
        # Numeric features
        for col in self._numeric_features:
            row.append(float(kwargs.get(col, 0)))

        # Categorical features — encode to ordinal codes
        for col in self._categorical_features:
            val = str(kwargs.get(col, "none"))
            code = self._cat_encoders[col].get(val, -1)  # -1 for unknown
            row.append(float(code))

        # Must match the dtype from training (DataFrame .values with mixed types → object)
        return np.array([row], dtype=object)

    def _sample_from_quantiles(self, q_vals: list) -> float:
        """Sample a yard value by interpolating between predicted quantiles."""
        u = random.random()
        q10, q90 = q_vals[0], q_vals[-1]
        iqr = q90 - q10
        extended_q = [0.0] + self._quantile_levels + [1.0]
        extended_v = [q10 - iqr * 0.5] + q_vals + [q90 + iqr * 0.5]
        return float(np.interp(u, extended_q, extended_v))

    def predict(self, **kwargs) -> tuple[str, float]:
        """Predict play outcome and yards gained.

        Returns:
            (outcome, yards) where outcome is one of
            'normal', 'touchdown', 'interception', 'fumble_lost'
        """
        X = self._encode_features(**kwargs)

        # Stage 1: sample outcome from predicted probabilities
        proba = self._clf.predict_proba(X)[0]
        outcome_idx = np.random.choice(len(self._outcome_mapping), p=proba)
        outcome = self._outcome_mapping[str(outcome_idx)]

        # Stage 2: sample yards given outcome
        info = self._regressors[outcome]
        if info["type"] == "quantile":
            q_vals = [info["models"][q].predict(X)[0] for q in self._quantile_levels]
            yards = self._sample_from_quantiles(q_vals)
        else:
            yards = random.choice(info["yards"])

        return outcome, yards

    def predict_proba(self, **kwargs) -> dict[str, float]:
        """Return outcome probabilities without sampling."""
        X = self._encode_features(**kwargs)
        proba = self._clf.predict_proba(X)[0]
        return {self._outcome_mapping[str(i)]: float(p) for i, p in enumerate(proba)}

    def predict_quantiles(self, outcome: str, **kwargs) -> dict[float, float]:
        """Return predicted quantile values for a given outcome."""
        X = self._encode_features(**kwargs)
        info = self._regressors[outcome]
        if info["type"] == "quantile":
            return {q: float(info["models"][q].predict(X)[0]) for q in self._quantile_levels}
        else:
            yards = info["yards"]
            return {q: float(np.percentile(yards, q * 100)) for q in self._quantile_levels}
