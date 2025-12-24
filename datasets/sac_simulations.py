import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pycox.simulations import SimStudySACCensorConst


class SimSACConstTrainTestData:
    """
    Simulated survival dataset using pycox.simulations.SimStudySACCensorConst.

    Official split:
      - Training: generated separately with size n_train, cached as train_{n_train}.csv
      - Testing: fixed size n_test, cached as test_{n_test}.csv

    CSV schema:
      - covariates: x0 ... x44
      - duration: float, in MONTHS originally -> saved/used in DAYS (months * 30.4375)
      - event: int, 1=observed event, 0=censored

    This class is designed to be called by the same main function as RotterdamGBSGData:
      - constructor exposes step/seed/pad_left/pad_right
      - get_official_train_test() returns (train_dataset, test_dataset)
      - SurvivalDataset yields dict with keys: data, label, duration, event
    """

    def __init__(self, root, n_train, n_test, step=1.0, seed=42, pad_left=1, pad_right=1):
        self.root = root
        self.n_train = int(n_train)
        self.n_test = int(n_test)
        self.step = float(step)
        self.seed = int(seed)
        self.pad_left = int(pad_left)
        self.pad_right = int(pad_right)

        os.makedirs(self.root, exist_ok=True)

        # Ensure cached files exist (generate once, then reuse)
        self.train_path = self._train_path(self.n_train)
        self.test_path = self._test_path(self.n_test)

        if not os.path.exists(self.train_path):
            self._generate_and_save(split="train", n=self.n_train, path=self.train_path, seed=self.seed + 0)
        if not os.path.exists(self.test_path):
            self._generate_and_save(split="test", n=self.n_test, path=self.test_path, seed=self.seed + 1)

        # Load cached splits
        self.train_df = self._load_csv(self.train_path)
        self.test_df = self._load_csv(self.test_path)

        # Extract arrays
        self.train_data, self.train_duration, self.train_event = self._df_to_arrays(self.train_df)
        self.test_data, self.test_duration, self.test_event = self._df_to_arrays(self.test_df)

        # Dataset properties (use training split for n_features)
        self.n_features = self.train_data.shape[1]  # expected 45

        # n_events: ignore censoring class 0, consistent with your RotterdamGBSGData logic
        self.n_events = int(
            max(
                len(np.unique(self.train_event)) - 1,
                len(np.unique(self.test_event)) - 1,
            )
        )

        # Discretize durations (durations are already in DAYS in the cached CSVs)
        self.train_label = self._duration_to_label(self.train_duration)
        self.test_label = self._duration_to_label(self.test_duration)

        # n_classes: based on max label across both splits + padding on both ends
        max_label = max(self.train_label.max(), self.test_label.max())
        self.n_classes = int(max_label + self.pad_left + self.pad_right)

    # ---------- paths ----------
    def _train_path(self, n_train: int) -> str:
        return os.path.join(self.root, f"train_{int(n_train)}.csv")

    def _test_path(self, n_test: int) -> str:
        return os.path.join(self.root, f"test_{int(n_test)}.csv")

    # ---------- schema helpers ----------
    @staticmethod
    def _feature_cols():
        return [f"x{i}" for i in range(45)]

    @classmethod
    def _required_cols(cls):
        return cls._feature_cols() + ["duration", "event"]

    def _validate_df(self, df: pd.DataFrame, context: str):
        missing = set(self._required_cols()) - set(df.columns)
        if missing:
            raise ValueError(f"[{context}] Missing columns: {sorted(missing)}")

    # ---------- generation / IO ----------
    def _generate_and_save(self, split: str, n: int, path: str, seed: int):
        """
        Generate a split once and save it. We do not attempt to guarantee reproducibility
        beyond caching: once saved, we always reuse the same file.
        """
        # Best-effort seed control (even if pycox uses other RNGs, caching is the real guarantee)
        np.random.seed(int(seed))

        sim = SimStudySACCensorConst()
        data = sim.simulate(int(n))
        df = sim.dict2df(data, True, False)

        self._validate_df(df, context=f"generated-{split}")

        # Hard-coded months -> days scaling (as requested)
        df = df.copy()
        df["duration"] = df["duration"].astype(np.float32) * 30.4375
        df["event"] = df["event"].astype(np.int64)  # event=1 observed, 0 censored

        # Ensure covariate dtypes
        for c in self._feature_cols():
            df[c] = df[c].astype(np.float32)

        # Save
        df.to_csv(path, index=False)

    def _load_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        self._validate_df(df, context=f"load:{os.path.basename(path)}")

        # Coerce dtypes
        df = df.copy()
        for c in self._feature_cols():
            df[c] = df[c].astype(np.float32)
        df["duration"] = df["duration"].astype(np.float32)  # assumed already in DAYS (cached that way)
        df["event"] = df["event"].astype(np.int64)
        return df

    def _df_to_arrays(self, df: pd.DataFrame):
        X = df[self._feature_cols()].values.astype(np.float32)
        duration = df["duration"].values.astype(np.float32)
        event = df["event"].values.astype(np.float32)
        return X, duration, event

    # ---------- discretization ----------
    def _duration_to_label(self, duration):
        bin_idx = (duration // self.step).astype(np.int64) + self.pad_left
        return bin_idx

    # ---------- public API ----------
    def get_official_train_test(self):
        train_dataset = SurvivalDataset(
            parent_data=self,
            data=self.train_data,
            duration=self.train_duration,
            event=self.train_event,
            label=self.train_label,
            n_features=self.n_features,
            n_classes=self.n_classes,
            n_events=self.n_events,
        )

        test_dataset = SurvivalDataset(
            parent_data=self,
            data=self.test_data,
            duration=self.test_duration,
            event=self.test_event,
            label=self.test_label,
            n_features=self.n_features,
            n_classes=self.n_classes,
            n_events=self.n_events,
        )
        return train_dataset, test_dataset


class SurvivalDataset(Dataset):
    def __init__(self, parent_data, data, duration, event, label, n_features, n_classes, n_events):
        # No train_ratio subsampling (as requested)
        self.data = data.astype(np.float32)
        self.duration = duration.astype(np.int64)
        self.event = event.astype(np.int64)
        self.label = label.astype(np.int64)
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_events = n_events
        self._duration_to_label = parent_data._duration_to_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "data": self.data[idx],
            "label": self.label[idx],
            "duration": self.duration[idx],
            "event": self.event[idx],
        }


if __name__ == "__main__":
    # Example usage (same pattern as your RotterdamGBSGData)
    data_loader = SimSACConstTrainTestData(root="./sim_sac", n_train=5000, step=30.0, seed=42)
    train_dataset, test_dataset = data_loader.get_official_train_test()

    print(f"Training set size: {len(train_dataset)}")  # 5000
    print(f"Test set size: {len(test_dataset)}")       # 10000
    print(f"Number of features: {train_dataset.n_features}")
    print(f"Number of classes: {train_dataset.n_classes}")
    print(f"Number of events: {train_dataset.n_events}")

    sample = train_dataset[0]
    print(sample.keys(), sample["data"].shape)
