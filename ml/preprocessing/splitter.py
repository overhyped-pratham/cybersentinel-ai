"""
CyberSentinel AI - Scenario-Based Dataset Splitter.

Enforces Rule 5: NEVER randomly split individual temporal rows when doing so leaks
future context or temporal autocorrelation.
Always splits by discrete scenario traces / session IDs.
"""

from typing import List, Tuple, Dict, Optional, Set
import numpy as np
import pandas as pd


class ScenarioBasedSplitter:
    """
    Splits network states or sequences into Train, Validation, and Test
    partitions strictly by scenario/trace identifier.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ) -> None:
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0; got {total}")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

    def split_scenarios(
        self,
        scenario_ids: List[str],
        custom_groups: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, List[str]]:
        """
        Partitions scenario IDs into train, val, and test lists.
        """
        unique_scenarios = sorted(list(set(scenario_ids)))
        if not unique_scenarios:
            return {"train": [], "val": [], "test": []}

        if custom_groups:
            train_sc = custom_groups.get("train", [])
            val_sc = custom_groups.get("val", [])
            test_sc = custom_groups.get("test", [])
            self.verify_no_overlap(train_sc, val_sc, test_sc)
            return {"train": train_sc, "val": val_sc, "test": test_sc}

        rng = np.random.default_rng(self.random_seed)
        shuffled = list(unique_scenarios)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = max(1, int(round(n * self.train_ratio)))
        n_val = int(round(n * self.val_ratio))
        if n > 2 and n_val == 0:
            n_val = 1

        train_sc = shuffled[:n_train]
        val_sc = shuffled[n_train:n_train + n_val]
        test_sc = shuffled[n_train + n_val:]

        # Handle edge cases where test might be empty due to rounding
        if not test_sc and len(train_sc) > 1:
            test_sc = [train_sc.pop()]

        self.verify_no_overlap(train_sc, val_sc, test_sc)
        return {"train": train_sc, "val": val_sc, "test": test_sc}

    def split_dataframe(
        self,
        df: pd.DataFrame,
        custom_groups: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits a DataFrame containing a scenario_id column into train, val, and test DataFrames.
        """
        if "scenario_id" not in df.columns:
            raise ValueError("DataFrame must contain 'scenario_id' column for scenario-based splitting.")

        all_scenarios = df["scenario_id"].unique().tolist()
        splits = self.split_scenarios(all_scenarios, custom_groups=custom_groups)

        df_train = df[df["scenario_id"].isin(splits["train"])].copy().reset_index(drop=True)
        df_val = df[df["scenario_id"].isin(splits["val"])].copy().reset_index(drop=True)
        df_test = df[df["scenario_id"].isin(splits["test"])].copy().reset_index(drop=True)

        self.verify_no_overlap(
            df_train["scenario_id"].unique().tolist(),
            df_val["scenario_id"].unique().tolist(),
            df_test["scenario_id"].unique().tolist(),
        )

        return df_train, df_val, df_test

    @staticmethod
    def verify_no_overlap(
        train_scenarios: List[str],
        val_scenarios: List[str],
        test_scenarios: List[str],
    ) -> None:
        """
        Asserts that train, val, and test sets are strictly disjoint.
        Raises AssertionError if leakage is detected.
        """
        set_train = set(train_scenarios)
        set_val = set(val_scenarios)
        set_test = set(test_scenarios)

        train_val_leak = set_train.intersection(set_val)
        train_test_leak = set_train.intersection(set_test)
        val_test_leak = set_val.intersection(set_test)

        if train_val_leak or train_test_leak or val_test_leak:
            raise AssertionError(
                f"CRITICAL LEAKAGE DETECTED across scenario splits!\n"
                f"Train-Val overlap: {train_val_leak}\n"
                f"Train-Test overlap: {train_test_leak}\n"
                f"Val-Test overlap: {val_test_leak}"
            )
