import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import KFold, StratifiedKFold


class METABRICData:
    def __init__(self, feature_file, label_file, step=1.0, stratify=False, 
                 kfold=5, seed=42, unit_scale=1.0, pad_left=1, pad_right=1):

        self.data, self.duration, self.event = self._load_and_normalize(feature_file, label_file)
        self.n_features = self.data.shape[1]
        self.n_events = int(len(np.unique(self.event)) - 1)  # ignore censoring
        self.stratify = stratify
        self.kfold = kfold
        self.seed = seed
        self.step = step
        self.unit_scale = unit_scale
        self.pad_left = pad_left
        self.pad_right = pad_right

        # this merabric dataset from deephit seems to have durations in the unit of days
        # thus no need to scale it
        self.duration = (self.duration * unit_scale).astype(np.float32)
        self.label = self._duration_to_label(self.duration)
        self.n_classes = int(self.label.max() + self.pad_left + self.pad_right)  # including padding at both ends
        

    def _load_and_normalize(self, feature_file, label_file):
        df_data = pd.read_csv(feature_file)
        df_label = pd.read_csv(label_file)

        data = df_data.values.astype(np.float32)
        duration = df_label['event_time'].values.astype(np.float32)
        event = df_label['label'].values.astype(np.float32)

        # data = self._normalize(data)
        return data, duration, event

    def _normalize(self, X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1
        return (X - mean) / std

    def _duration_to_label(self, duration):
        bin_idx = (duration // self.step).astype(np.int64) + self.pad_left  # padding on the left
        return bin_idx

    def get_kfold_datasets(self):
        if self.stratify:
            kf = StratifiedKFold(n_splits=self.kfold, shuffle=True, random_state=self.seed)
            for train_idx, test_idx in kf.split(self.data, self.event):
                yield METABRICDataset(self, train_idx), METABRICDataset(self, test_idx)
        else:
            kf = KFold(n_splits=self.kfold, shuffle=True, random_state=self.seed)
            for train_idx, test_idx in kf.split(self.data):
                yield METABRICDataset(self, train_idx), METABRICDataset(self, test_idx)


class METABRICDataset(Dataset):
    def __init__(self, metabric_data, indices):
        self.data = metabric_data.data[indices].astype(np.float32)
        self.duration = metabric_data.duration[indices].astype(np.int64)
        self.event = metabric_data.event[indices].astype(np.int64)
        self.label = metabric_data.label[indices].astype(np.int64)
        self.n_features = metabric_data.n_features
        self.n_classes = metabric_data.n_classes
        self.n_events = metabric_data.n_events
        self._duration_to_label = metabric_data._duration_to_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'data': self.data[idx],
            'label': self.label[idx],
            'duration': self.duration[idx],
            'event': self.event[idx],
        }

