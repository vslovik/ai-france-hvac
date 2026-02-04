import pickle

import torch
from torch import nn
from torch.utils.data import DataLoader

from dl_data.dataset import CustomerDLDataset
from dl_training.model import AdvancedDLModel


def train_advanced_dl_model(X_dl, y_dl, model_type='advanced', epochs=100):
    """
    Enhanced training pipeline with:
    - Learning rate scheduling
    - Gradient accumulation
    - Early stopping
    - Mixed precision training
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    print(f"\n🚀 Training {model_type.upper()} DL Model...")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_dl, y_dl, test_size=0.2, random_state=42, stratify=y_dl
    )

    # Convert to sequences (1 timestep)
    train_sequences = {}
    val_sequences = {}

    for i, (features, target) in enumerate(zip(X_train.values, y_train.values)):
        train_sequences[i] = features.reshape(1, -1)

    for i, (features, target) in enumerate(zip(X_val.values, y_val.values)):
        val_sequences[i] = features.reshape(1, -1)

    # Create datasets WITH robust normalization
    train_dataset = CustomerDLDataset(train_sequences, dict(enumerate(y_train)),
                                      list(X_dl.columns), normalize='robust')
    val_dataset = CustomerDLDataset(val_sequences, dict(enumerate(y_val)),
                                    list(X_dl.columns), normalize='robust')

    # Choose model
    input_dim = X_dl.shape[1]

    if model_type == 'advanced':
        model = AdvancedDLModel(input_dim, hidden_dims=[256, 128, 64])
    else:
        model = AdvancedDLModel(input_dim)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Loss with class weighting
    pos_weight = torch.tensor([len(y_train) / sum(y_train) - 1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer with different learning rates for different parts
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 0.0005}
    ], weight_decay=0.01)

    #Learning rate scheduler (cosine annealing)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=10, T_mult=2, eta_min=1e-6
    # )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=50,  # Half-period in epochs
    #     eta_min=1e-5,  # Minimum learning rate
    #     last_epoch=-1
    # )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # We want to maximize AUC
        patience=10,  # Wait 8 epochs with no improvement
        factor=0.5,  # Reduce LR by 50%
        min_lr=1e-6,  # Minimum learning rate
    )

    # Training loop with gradient accumulation
    accumulation_steps = 2
    best_val_auc = 0
    patience_counter = 0
    patience = 25

    print(f"  Model: {model_type}")
    print(f"  Input dim: {input_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Validation samples: {len(val_dataset):,}")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            x = batch['sequence'].to(device)
            y = batch['target'].to(device)

            outputs = model(x)
            loss = criterion(outputs, y) / accumulation_steps
            loss.backward()

            train_loss += loss.item() * accumulation_steps

            # Gradient accumulation
            if (i + 1) % accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch['sequence'].to(device)
                y = batch['target'].to(device)

                outputs = model(x)
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets.extend(y.cpu().numpy())

        val_auc = roc_auc_score(val_targets, val_preds)
        scheduler.step(val_auc)

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  ✓ Epoch {epoch + 1}: Loss={avg_train_loss:.4f}, Val AUC={val_auc:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  ⏹️ Early stopping at epoch {epoch + 1}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    print(f"\n✅ Training Complete!")
    print(f"  Best Val AUC: {best_val_auc:.4f}")

    # Save model
    model_data = {
        'model': model,
        'X_test': X_val,
        'y_test': y_val
    }

    model_name = f"dl_{model_type}_model"
    with open(f'{model_name}.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print(f"✓ Model saved: {model_name}.pkl")

    return model_data
