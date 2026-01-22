def create_sequences(X, y):
    """
    Convert RF's tabular features to DL sequences
    Each sequence observation becomes a 1-timestep sequence
    """
    print("\n" + "=" * 80)
    print("CONVERTING RF FEATURES TO DL SEQUENCES")
    print("=" * 80)

    # Convert to DL sequence format (1 timestep per observation)
    sequences = {}
    targets = {}

    for idx in range(len(X)):
        sequence = X.iloc[idx].values.reshape(1, -1)  # Shape: [1, features]
        sequences[idx] = sequence
        targets[idx] = y.iloc[idx]

    return sequences, targets, X.columns.tolist()