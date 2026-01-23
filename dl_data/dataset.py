import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class CustomerDLDataset(Dataset):
    """
    Dataset that uses RF's features for DL training
    """

    def __init__(self, sequences, targets, feature_names, normalize=True):
        self.sequences = sequences
        self.targets = targets
        self.indices = list(sequences.keys())
        self.feature_names = feature_names

        # Get sequence dimensions
        sample_seq = sequences[self.indices[0]]
        self.seq_length = sample_seq.shape[0]  # Should be 1
        self.feature_dim = sample_seq.shape[1]
        if normalize:
            print("🔧 Normalizing features for DL...")
            self.sequences = self._normalize_sequences(sequences)
            print(f"  Normalization complete!")
        # Number of features

    def _normalize_sequences(self, sequences):
        """Normalize all sequences to mean=0, std=1"""
        # Collect all features
        all_features = []
        for seq in sequences.values():
            all_features.append(seq)
        all_features = np.vstack(all_features)

        # Compute mean and std
        self.feature_mean = all_features.mean(axis=0)
        self.feature_std = all_features.std(axis=0)

        # Avoid division by zero
        self.feature_std = np.where(self.feature_std < 1e-8, 1.0, self.feature_std)

        print(f"  Before: min={all_features.min():.2f}, max={all_features.max():.2f}")
        print(f"  Before: mean={all_features.mean():.2f}, std={all_features.std():.2f}")

        # Normalize sequences
        normalized_sequences = {}
        for idx, seq in sequences.items():
            normalized_seq = (seq - self.feature_mean) / self.feature_std
            normalized_sequences[idx] = normalized_seq

            # Debug: Check first sequence
            if idx == 0:
                print(f"  After: min={normalized_seq.min():.2f}, max={normalized_seq.max():.2f}")
                print(f"  After: mean={normalized_seq.mean():.2f}, std={normalized_seq.std():.2f}")

        return normalized_sequences

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        seq_idx = self.indices[idx]
        sequence = self.sequences[seq_idx]
        target = self.targets[seq_idx]

        return {
            'sequence': torch.FloatTensor(sequence),
            'target': torch.FloatTensor([target]),
            'seq_length': torch.LongTensor([self.seq_length])
        }

    def get_stats(self):
        return {
            'num_samples': len(self),
            'seq_length': self.seq_length,
            'feature_dim': self.feature_dim,
            'conversion_rate': np.mean(list(self.targets.values())),
            'feature_names': self.feature_names[:10]  # First 10 features
        }
