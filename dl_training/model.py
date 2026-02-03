import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDLModel(nn.Module):
    """
    Simple DL model that uses RF's features
    Should match or beat RF's 0.8046 AUC
    """

    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Input shape: [batch, seq_len, features]
        # Since seq_len=1, squeeze the sequence dimension
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)  # [batch, features]

        return self.model(x)


class AdvancedDLModel(nn.Module):
    """Attention on INPUT features, not hidden features"""

    def __init__(self, n_features, hidden_dims=256, dropout_rate=0.3):
        super().__init__()

        if isinstance(hidden_dims, list):
            hidden_size = hidden_dims[0]
        else:
            hidden_size = hidden_dims

        # ATTENTION ON INPUT FEATURES
        self.input_attention = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_features),  # Output per-input-feature weights
            nn.Sigmoid()
        )

        # Main network
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(hidden_size),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(hidden_size // 2),

            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate - 0.1),
            nn.BatchNorm1d(hidden_size // 4),
        )

        self.classifier = nn.Linear(hidden_size // 4, 1)
        self._initialize_weights()

        print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        if isinstance(x, dict) and 'sequence' in x:
            x = x['sequence']
        if x.dim() == 3:
            x = x.squeeze(1)

        # Apply attention to INPUT features
        input_attention = self.input_attention(x)  # [batch, n_features]
        x_weighted = x * input_attention

        # Process through network
        features = self.net(x_weighted)
        logits = self.classifier(features)

        return logits