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


class AdvancedDLModel1(nn.Module):
    """
    Simple but effective version - no dimension mismatches
    """

    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], dropout_rate=0.3):
        super().__init__()

        print(f"\n🎯 SimpleSuperiorDLModel with {input_dim} input features")

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        )

        print(f"  Total parameters: {self.count_parameters():,}")

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)

        x = self.hidden_layers(x)
        return self.output(x)


class AdvancedDLModel2(nn.Module):
    """
    Superior DL model compatible with your existing training pipeline
    """

    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], dropout_rate=0.3):
        super().__init__()

        print(f"\n🎯 SuperiorDLModel with {input_dim} input features")

        # SIMPLER ARCHITECTURE - no skip connection issues
        layers = []
        prev_dim = input_dim

        # Input batch norm
        self.input_bn = nn.BatchNorm1d(input_dim)

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        # Output with confidence calibration
        self.output = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Initialize weights
        self._initialize_weights()

        print(f"  Total parameters: {self.count_parameters():,}")

    def _initialize_weights(self):
        """Proper weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        """
        Forward pass that handles different input formats

        Your training code might pass:
        1. A tensor directly
        2. A dictionary with 'sequence' key
        3. A batch with extra dimensions
        """
        # DEBUG: Print what we're getting
        # print(f"DEBUG - Input type: {type(x)}")
        # if isinstance(x, dict):
        #     print(f"DEBUG - Dict keys: {x.keys()}")
        # else:
        #     print(f"DEBUG - Tensor shape: {x.shape}")

        # Handle dictionary input (common in sequence models)
        if isinstance(x, dict) and 'sequence' in x:
            x = x['sequence']

        # Remove sequence dimension if present (B, 1, F) -> (B, F)
        if x.dim() == 3:
            x = x.squeeze(1)

        # Input normalization
        x = self.input_bn(x)

        # Hidden layers
        x = self.hidden_layers(x)

        # Output
        return self.output(x)


class AdvancedDLModel3(nn.Module):
    def __init__(self, n_features, hidden_dims=256, dropout_rate=0.6):
        super().__init__()
        self.n_features = n_features

        # Handle both integer and list inputs
        if isinstance(hidden_dims, list):
            hidden_size = hidden_dims[0]
            print(f"⚠️ Using first value from hidden_dims list: {hidden_size}")
        else:
            hidden_size = hidden_dims

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

        self.attention = nn.Sequential(
            nn.Linear(hidden_size // 4, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Linear(hidden_size // 4, 1)
        self._initialize_weights()

        # ADD THIS METHOD:
        total_params = self.count_parameters()
        print(f"  Parameters: {total_params:,}")

    # ADD THIS METHOD
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        """Simplified forward that returns logits with correct shape"""
        # Handle input format
        if isinstance(x, dict) and 'sequence' in x:
            x = x['sequence']
        if x.dim() == 3:
            x = x.squeeze(1)

        features = self.net(x)
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        logits = self.classifier(attended_features)

        # Return logits with shape [batch, 1] not [batch]
        return logits


class AdvancedDLModel4(nn.Module):
    """Your credit model adapted for France data - ACCEPTS LIST"""

    def __init__(self, n_features, hidden_dims=256, dropout_rate=0.3):
        super().__init__()

        # Handle both integer and list inputs
        if isinstance(hidden_dims, list):
            print(f"  Hidden dims (list): {hidden_dims}")
            # If list, use the first value for this architecture
            hidden_size = hidden_dims[0]
        else:
            print(f"  Hidden dims (int): {hidden_dims}")
            hidden_size = hidden_dims

        # Your exact credit architecture
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

        self.attention = nn.Sequential(
            nn.Linear(hidden_size // 4, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Linear(hidden_size // 4, 1)
        self._initialize_weights()

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Parameters: {total_params:,}")

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

        features = self.net(x)
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        logits = self.classifier(attended_features)

        return logits


class AdvancedDLModelW(nn.Module):
    """Fixed version with proper attention"""

    def __init__(self, n_features, hidden_dims=256, dropout_rate=0.3):
        super().__init__()

        # Handle both integer and list inputs
        if isinstance(hidden_dims, list):
            print(f"  Hidden dims (list): {hidden_dims}")
            hidden_size = hidden_dims[0]
        else:
            print(f"  Hidden dims (int): {hidden_dims}")
            hidden_size = hidden_dims

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

        # FIXED ATTENTION: Output size matches feature size
        final_features = hidden_size // 4
        self.attention = nn.Sequential(
            nn.Linear(final_features, 64),  # Intermediate layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, final_features),  # Output per-feature weights!
            nn.Sigmoid()  # Each feature gets its own weight [0, 1]
        )

        self.classifier = nn.Linear(final_features, 1)
        self._initialize_weights()

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Parameters: {total_params:,}")

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

        features = self.net(x)  # Shape: [batch, hidden_size//4]

        # FIXED: Attention outputs weights for EACH feature
        attention_weights = self.attention(features)  # Shape: [batch, hidden_size//4]

        attended_features = features * attention_weights  # Element-wise
        logits = self.classifier(attended_features)

        return logits


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