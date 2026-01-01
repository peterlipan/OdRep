import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional, Tuple, Dict, Type


class SurvivalDataGenerator:
    """Base class for survival data simulation with censoring."""
    
    def __init__(self, n: int, d: int, seed: int = 0):
        self.n = n
        self.d = d
        self.rng = np.random.default_rng(seed)
    
    def _make_beta(self, n_nonzero: int = 10, scale: float = 0.5) -> np.ndarray:
        """Create sparse coefficient vector."""
        beta = np.zeros(self.d, dtype=float)
        idx = self.rng.choice(self.d, size=min(n_nonzero, self.d), replace=False)
        beta[idx] = self.rng.normal(loc=0.0, scale=scale, size=len(idx))
        return beta
    
    def _make_covariates(self, x_dist: str = "normal") -> np.ndarray:
        """Generate covariate matrix."""
        if x_dist == "normal":
            return self.rng.normal(size=(self.n, self.d))
        elif x_dist == "uniform":
            return self.rng.uniform(low=-1.0, high=1.0, size=(self.n, self.d))
        raise ValueError("x_dist must be 'normal' or 'uniform'")
    
    def _apply_censoring(self, T: np.ndarray, target_rate: float, 
                        max_iter: int = 40) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Apply exponential censoring to achieve target censoring rate.
        Returns: (observed_times, event_indicators, censoring_rate_used)
        """
        lo, hi = -12.0, 6.0
        best_rate = None
        best_err = float('inf')
        
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            rate = np.exp(mid)
            
            C_test = self.rng.exponential(scale=1.0 / rate, size=len(T))
            censor_frac = np.mean(C_test < T)
            err = abs(censor_frac - target_rate)
            
            if err < best_err:
                best_err = err
                best_rate = rate
            
            if censor_frac < target_rate:
                lo = mid
            else:
                hi = mid
        
        C = self.rng.exponential(scale=1.0 / best_rate, size=len(T))
        Y = np.minimum(T, C)
        E = (T <= C).astype(int)
        
        return Y, E, best_rate
    
    def _make_dataframe(self, X: np.ndarray, Y: np.ndarray, E: np.ndarray,
                       extra_cols: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
        """Construct output DataFrame."""
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(self.d)])
        df["duration"] = Y
        df["event"] = E
        if extra_cols:
            for name, values in extra_cols.items():
                df[name] = values
        return df


class ProportionalHazardsSimulator(SurvivalDataGenerator):
    """Simulate PH model with Weibull baseline: S(t|x) = exp(-((t/λ)^k) * exp(x^T β))"""
    
    def generate(self, beta: Optional[np.ndarray] = None, n_nonzero: int = 10,
                beta_scale: float = 0.5, x_dist: str = "normal",
                baseline_shape: float = 1.5, baseline_scale: float = 10.0,
                target_censor: float = 0.4) -> pd.DataFrame:
        
        X = self._make_covariates(x_dist)
        
        if beta is None:
            beta = self._make_beta(n_nonzero, beta_scale)
        beta = np.asarray(beta, dtype=float).reshape(-1)
        assert len(beta) == self.d
        
        r = X @ beta
        U = self.rng.uniform(low=1e-12, high=1.0 - 1e-12, size=self.n)
        T = baseline_scale * ((-np.log(U) / np.exp(r)) ** (1.0 / baseline_shape))
        
        Y, E, _ = self._apply_censoring(T, target_censor)
        return self._make_dataframe(X, Y, E)


class ProportionalOddsSimulator(SurvivalDataGenerator):
    """Simulate PO model with log-logistic baseline."""
    
    def generate(self, beta: Optional[np.ndarray] = None, n_nonzero: int = 10,
                beta_scale: float = 0.5, x_dist: str = "normal",
                baseline_shape: float = 1.5, baseline_scale: float = 10.0,
                target_censor: float = 0.4) -> pd.DataFrame:
        
        X = self._make_covariates(x_dist)
        
        if beta is None:
            beta = self._make_beta(n_nonzero, beta_scale)
        beta = np.asarray(beta, dtype=float).reshape(-1)
        assert len(beta) == self.d
        
        r = X @ beta
        U = self.rng.uniform(low=1e-12, high=1.0 - 1e-12, size=self.n)
        er = np.exp(r)
        
        q = U / (er * (1.0 - U) + U)
        q = np.clip(q, 1e-12, 1.0 - 1e-12)
        
        T = baseline_scale * ((q / (1.0 - q)) ** (1.0 / baseline_shape))
        
        Y, E, _ = self._apply_censoring(T, target_censor)
        return self._make_dataframe(X, Y, E)


class StochasticMonotoneLink:
    """
    Random monotone inverse-link: h*(z) = sigmoid(φ(z))
    where φ(z) = b + αz + Σ w_l * softplus(z - t_l) is monotone by construction.
    """
    
    def __init__(self, z_min: float = -8.0, z_max: float = 8.0, L_star: int = 12,
                 alpha_range: Tuple[float, float] = (0.5, 1.5), w_scale: float = 1.0,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.z_min = z_min
        self.z_max = z_max
        
        self.b = rng.normal(0.0, 0.5)
        self.alpha = rng.uniform(*alpha_range)
        self.t = np.sort(rng.uniform(z_min, z_max, size=L_star))
        self.w = rng.lognormal(mean=np.log(max(w_scale, 1e-6)), sigma=0.6, size=L_star)
    
    def _softplus(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -50, 50)
        return np.log1p(np.exp(x))
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(-x))
    
    def phi(self, z: np.ndarray) -> np.ndarray:
        """Monotone transformation before sigmoid."""
        zc = np.clip(z, self.z_min, self.z_max)
        ramps = self._softplus(zc[..., None] - self.t[None, :])
        return self.b + self.alpha * zc + ramps @ self.w
    
    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self._sigmoid(self.phi(z))


class DiscreteTimeLinkSimulator(SurvivalDataGenerator):
    """
    Discrete-time survival with stochastic monotone link for testing link recovery methods.
    DGP: F_k(x) = h*(η_k + x^T β) where h* is a random monotone function.
    """
    
    def generate(self, m: int = 50, tau_max: float = 20.0,
                taus: Optional[np.ndarray] = None, beta: Optional[np.ndarray] = None,
                n_nonzero: int = 10, beta_scale: float = 0.6, x_dist: str = "normal",
                target_censor: float = 0.4, z_min: float = -8.0, z_max: float = 8.0,
                L_star: int = 12) -> pd.DataFrame:
        
        if taus is None:
            taus = np.linspace(0.0, tau_max, m + 1)
        else:
            taus = np.asarray(taus, dtype=float)
            m = len(taus) - 1
        assert np.all(np.diff(taus) > 0), "taus must be strictly increasing"
        
        X = self._make_covariates(x_dist)
        
        if beta is None:
            beta = self._make_beta(n_nonzero, beta_scale)
        beta = np.asarray(beta, dtype=float).reshape(-1)
        assert len(beta) == self.d
        
        r = X @ beta
        
        t = np.linspace(1.0 / (m + 2), (m + 1.0) / (m + 2), m)
        eta = np.log(t / (1.0 - t))
        eta = (eta - eta.mean()) / (eta.std() + 1e-8)
        eta = np.clip(2.0 * eta, z_min, z_max)
        
        link = StochasticMonotoneLink(z_min, z_max, L_star, seed=self.rng.integers(1e9))
        
        Z = r[:, None] + eta[None, :]
        F = link(Z)
        F = np.maximum.accumulate(F, axis=1)
        F = np.clip(F, 1e-8, 1.0 - 1e-8)
        
        F_prev = np.concatenate([np.zeros((self.n, 1)), F[:, :-1]], axis=1)
        pmf_bins = np.clip(F - F_prev, 1e-12, None)
        tail = np.clip(1.0 - F[:, -1], 1e-12, None)
        pmf = np.concatenate([pmf_bins, tail[:, None]], axis=1)
        pmf /= pmf.sum(axis=1, keepdims=True)
        
        K = np.array([self.rng.choice(m + 1, p=pmf[i]) for i in range(self.n)]) + 1
        
        T = np.empty(self.n, dtype=float)
        for i in range(self.n):
            k = K[i]
            if k <= m:
                T[i] = self.rng.uniform(taus[k - 1], taus[k])
            else:
                T[i] = taus[-1] + self.rng.exponential(scale=(taus[-1] - taus[-2]))
        
        Y, E, _ = self._apply_censoring(T, target_censor)
        
        return self._make_dataframe(X, Y, E)


# ============================================================================
# PyTorch Dataset Classes
# ============================================================================

class SurvivalPyTorchDataset(Dataset):
    """PyTorch Dataset wrapper for survival data."""
    
    def __init__(self, data: np.ndarray, duration: np.ndarray, event: np.ndarray, 
                 label: np.ndarray, n_features: int, n_classes: int, n_events: int, duration_to_label: None):
        self.data = data.astype(np.float32)
        self.duration = duration.astype(np.int64)
        self.event = event.astype(np.int64)
        self.label = label.astype(np.int64)
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_events = n_events
        self._duration_to_label = duration_to_label
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            "data": self.data[idx],
            "label": self.label[idx],
            "duration": self.duration[idx],
            "event": self.event[idx],
        }


class BaseSurvivalDataset:
    """
    Base class for generating, caching, and loading survival datasets.
    Handles data generation, CSV caching, validation, and discretization.
    """
    
    SIMULATOR_CLASS = None  # Override in child classes
    SIMULATOR_TYPE = None   # Override in child classes (for folder naming)
    DEFAULT_PARAMS = {}     # Override in child classes
    
    def __init__(self, root: str, n_train: int, n_test: int = 10000, d: int = 45,
                 step: float = 1.0, seed: int = 42, pad_left: int = 1, pad_right: int = 1):
        assert self.SIMULATOR_CLASS is not None, "Must override SIMULATOR_CLASS in child class"
        assert self.SIMULATOR_TYPE is not None, "Must override SIMULATOR_TYPE in child class"
        
        self.root = root
        self.n_train = int(n_train)
        self.n_test = int(n_test)
        self.d = int(d)
        self.step = float(step)
        self.seed = int(seed)
        self.pad_left = int(pad_left)
        self.pad_right = int(pad_right)
        
        # Create simulator-specific directory
        self.data_dir = os.path.join(self.root, self.SIMULATOR_TYPE)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Cache paths - include n_train and n_test in filename for uniqueness
        self.train_path = os.path.join(self.data_dir, f"train_{self.n_train}_test_{self.n_test}_seed_{self.seed}.csv")
        self.test_path = os.path.join(self.data_dir, f"test_{self.n_train}_test_{self.n_test}_seed_{self.seed}.csv")
        
        # Generate or load cached data (i.i.d. split from same generation)
        if not os.path.exists(self.train_path) or not os.path.exists(self.test_path):
            self._generate_and_split_save()
        
        # Load data
        self.train_df = self._load_csv(self.train_path)
        self.test_df = self._load_csv(self.test_path)
        
        # Extract arrays
        self.train_data, self.train_duration, self.train_event = self._df_to_arrays(self.train_df)
        self.test_data, self.test_duration, self.test_event = self._df_to_arrays(self.test_df)
        
        # Dataset properties
        self.n_features = self.d
        self.n_events = int(max(
            len(np.unique(self.train_event)) - 1,
            len(np.unique(self.test_event)) - 1,
        ))
        
        # Discretize durations
        self.train_label = self._duration_to_label(self.train_duration)
        self.test_label = self._duration_to_label(self.test_duration)
        
        # Calculate n_classes
        max_label = max(self.train_label.max(), self.test_label.max())
        self.n_classes = int(max_label + self.pad_left + self.pad_right)
    
    def _feature_cols(self):
        return [f"x{i}" for i in range(self.d)]
    
    def _required_cols(self):
        return self._feature_cols() + ["duration", "event"]
    
    def _validate_df(self, df: pd.DataFrame, context: str):
        missing = set(self._required_cols()) - set(df.columns)
        if missing:
            raise ValueError(f"[{context}] Missing columns: {sorted(missing)}")
    
    def _generate_and_split_save(self):
        """Generate full dataset with same seed/beta, then split into train/test."""
        n_total = self.n_train + self.n_test
        
        # Generate full dataset with single seed and beta
        simulator = self.SIMULATOR_CLASS(n=n_total, d=self.d, seed=self.seed)
        df_full = simulator.generate(**self.DEFAULT_PARAMS)
        
        self._validate_df(df_full, context="generated-full")
        
        # Ensure correct dtypes
        df_full = df_full.copy()
        for c in self._feature_cols():
            df_full[c] = df_full[c].astype(np.float32)
        df_full["duration"] = df_full["duration"].astype(np.float32)
        df_full["event"] = df_full["event"].astype(np.int64)
        
        # Split into train and test (first n_train for train, rest for test)
        train_df = df_full.iloc[:self.n_train].copy()
        test_df = df_full.iloc[self.n_train:].copy()
        
        # Save both splits
        train_df.to_csv(self.train_path, index=False)
        test_df.to_csv(self.test_path, index=False)
    
    def _load_csv(self, path: str) -> pd.DataFrame:
        """Load and validate cached CSV."""
        df = pd.read_csv(path)
        self._validate_df(df, context=f"load:{os.path.basename(path)}")
        
        df = df.copy()
        for c in self._feature_cols():
            df[c] = df[c].astype(np.float32)
        df["duration"] = df["duration"].astype(np.float32)
        df["event"] = df["event"].astype(np.int64)
        return df
    
    def _df_to_arrays(self, df: pd.DataFrame):
        """Extract numpy arrays from DataFrame."""
        X = df[self._feature_cols()].values.astype(np.float32)
        duration = df["duration"].values.astype(np.float32)
        event = df["event"].values.astype(np.float32)
        return X, duration, event
    
    def _duration_to_label(self, duration):
        """Discretize continuous durations into bins."""
        bin_idx = (duration // self.step).astype(np.int64) + self.pad_left
        return bin_idx
    
    def get_official_train_test(self):
        """Return train and test PyTorch datasets."""
        train_dataset = SurvivalPyTorchDataset(
            data=self.train_data,
            duration=self.train_duration,
            event=self.train_event,
            label=self.train_label,
            n_features=self.n_features,
            n_classes=self.n_classes,
            n_events=self.n_events,
            duration_to_label=self._duration_to_label,
        )
        
        test_dataset = SurvivalPyTorchDataset(
            data=self.test_data,
            duration=self.test_duration,
            event=self.test_event,
            label=self.test_label,
            n_features=self.n_features,
            n_classes=self.n_classes,
            n_events=self.n_events,
            duration_to_label=self._duration_to_label,
        )
        
        return train_dataset, test_dataset


class PHWeibullDataset(BaseSurvivalDataset):
    """Proportional Hazards with Weibull baseline dataset."""
    
    SIMULATOR_CLASS = ProportionalHazardsSimulator
    SIMULATOR_TYPE = "ph_weibull"
    DEFAULT_PARAMS = {
        'target_censor': 0.4,
        'baseline_shape': 1.5,
        'baseline_scale': 10.0,
        'n_nonzero': 10,
        'beta_scale': 0.5,
        'x_dist': 'normal',
    }


class POLogLogisticDataset(BaseSurvivalDataset):
    """Proportional Odds with log-logistic baseline dataset."""
    
    SIMULATOR_CLASS = ProportionalOddsSimulator
    SIMULATOR_TYPE = "po_loglogistic"
    DEFAULT_PARAMS = {
        'target_censor': 0.4,
        'baseline_shape': 1.5,
        'baseline_scale': 10.0,
        'n_nonzero': 10,
        'beta_scale': 0.5,
        'x_dist': 'normal',
    }


class LinkRecoveryDataset(BaseSurvivalDataset):
    """Discrete-time with stochastic monotone link dataset."""
    
    SIMULATOR_CLASS = DiscreteTimeLinkSimulator
    SIMULATOR_TYPE = "link_recovery"
    DEFAULT_PARAMS = {
        'm': 50,
        'tau_max': 20.0,
        'target_censor': 0.4,
        'n_nonzero': 10,
        'beta_scale': 0.6,
        'x_dist': 'normal',
        'z_min': -8.0,
        'z_max': 8.0,
        'L_star': 12,
    }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Test PH dataset
    ph_data = PHWeibullDataset(root="./sim_data", n_train=5000, n_test=10000, step=30.0, seed=42)
    ph_train, ph_test = ph_data.get_official_train_test()
    
    print(f"PH Training: {len(ph_train)} samples, {ph_train.n_features} features, {ph_train.n_classes} classes")
    print(f"PH Test: {len(ph_test)} samples")
    print(f"Sample: {ph_train[0].keys()}")
    
    # Test PO dataset
    po_data = POLogLogisticDataset(root="./sim_data", n_train=5000, n_test=10000, step=30.0, seed=42)
    po_train, po_test = po_data.get_official_train_test()
    
    print(f"\nPO Training: {len(po_train)} samples")
    
    # Test Link Recovery dataset
    link_data = LinkRecoveryDataset(root="./sim_data", n_train=5000, n_test=10000, step=30.0, seed=42)
    link_train, link_test = link_data.get_official_train_test()
    
    print(f"\nLink Recovery Training: {len(link_train)} samples")