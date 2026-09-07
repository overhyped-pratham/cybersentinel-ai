"""
CyberSentinel AI - Logistic Regression Baselines.

Implements the SIH-mandated classical baseline for:
1. Binary Attack Detection: S_t -> y_attack in {0, 1}
2. Multi-Class Current Stage Classification: S_t -> y_stage
3. Static Next-Stage Forecasting Baseline: S_t -> y_{t+1}

Serves as the static point-in-time benchmark to empirically determine whether
temporal sequencing (GRU) and world models provide measurable improvement.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression


class LogisticRegressionBaseline:
    """
    Configurable Logistic Regression baseline suite.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
        solver: str = "lbfgs",
    ) -> None:
        self.config = {
            "C": float(C),
            "max_iter": int(max_iter),
            "class_weight": class_weight,
            "random_state": int(random_state),
            "solver": solver,
        }
        self.attack_model = LogisticRegression(
            C=self.config["C"],
            max_iter=self.config["max_iter"],
            class_weight=self.config["class_weight"],
            random_state=self.config["random_state"],
            solver=self.config["solver"],
        )
        self.stage_model = LogisticRegression(
            C=self.config["C"],
            max_iter=self.config["max_iter"],
            class_weight=self.config["class_weight"],
            random_state=self.config["random_state"],
            solver=self.config["solver"],
        )
        self.next_stage_model = LogisticRegression(
            C=self.config["C"],
            max_iter=self.config["max_iter"],
            class_weight=self.config["class_weight"],
            random_state=self.config["random_state"],
            solver=self.config["solver"],
        )
        self.is_fitted = False
        self.stage_classes_: np.ndarray = np.array([])
        self.next_stage_classes_: np.ndarray = np.array([])
        self.attack_classes_: np.ndarray = np.array([])

    def fit(
        self,
        X_train: np.ndarray,
        y_attack_train: np.ndarray,
        y_stage_train: np.ndarray,
        y_next_stage_train: Optional[np.ndarray] = None,
    ) -> "LogisticRegressionBaseline":
        """
        Fits baseline models strictly on training splits.
        
        Args:
            X_train: Scaled feature matrix of shape (N, feature_dim).
            y_attack_train: Binary attack indicators (N,).
            y_stage_train: Integer stage labels (N,).
            y_next_stage_train: Optional integer next-stage targets (N,).
        """
        # Fit attack detector
        self.attack_model.fit(X_train, y_attack_train)
        self.attack_classes_ = self.attack_model.classes_

        # Fit current stage classifier
        self.stage_model.fit(X_train, y_stage_train)
        self.stage_classes_ = self.stage_model.classes_

        # Fit static next-stage predictor if next-stage labels provided
        if y_next_stage_train is not None:
            self.next_stage_model.fit(X_train, y_next_stage_train)
            self.next_stage_classes_ = self.next_stage_model.classes_

        self.is_fitted = True
        return self

    def predict_attack(self, X: np.ndarray) -> np.ndarray:
        """Predicts binary attack label: 0 (Benign) or 1 (Attack)."""
        self._check_fitted()
        return self.attack_model.predict(X)

    def predict_attack_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts attack probability array (N, 2)."""
        self._check_fitted()
        return self.attack_model.predict_proba(X)

    def predict_stage(self, X: np.ndarray) -> np.ndarray:
        """Predicts current attack stage ID."""
        self._check_fitted()
        return self.stage_model.predict(X)

    def predict_stage_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts probability distribution over stage classes."""
        self._check_fitted()
        return self.stage_model.predict_proba(X)

    def predict_next_stage(self, X: np.ndarray) -> np.ndarray:
        """Predicts next attack stage ID."""
        self._check_fitted()
        return self.next_stage_model.predict(X)

    def predict_next_stage_proba(self, X: np.ndarray, num_total_classes: Optional[int] = None) -> np.ndarray:
        """
        Predicts probability distribution over next stages.
        If num_total_classes is specified, maps probabilities into a full (N, num_total_classes) matrix.
        """
        self._check_fitted()
        probs = self.next_stage_model.predict_proba(X)
        if num_total_classes is None:
            return probs

        n_samples = X.shape[0]
        full_probs = np.zeros((n_samples, num_total_classes), dtype=float)
        for local_idx, class_id in enumerate(self.next_stage_classes_):
            if class_id < num_total_classes:
                full_probs[:, class_id] = probs[:, local_idx]
        return full_probs

    def save(self, file_path: Union[str, Path]) -> None:
        """Saves fitted baseline models to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "LogisticRegressionBaseline":
        """Loads fitted baseline from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("LogisticRegressionBaseline is not fitted. Call fit() first.")
