
def customer_split(df_quotes, train_size=0.95):
    print("\n" + "=" * 80)
    print(f"SPLIT CUSTOMERS: TRAIN vs SIMULATION: TRAINING SIZE {train_size}")
    print("=" * 80)

    cust_first = df_quotes.groupby('numero_compte')['dt_creation_devis'].min().reset_index()
    cust_first = cust_first.sort_values('dt_creation_devis')

    split_idx = int(len(cust_first) * train_size)
    split_date = cust_first.iloc[split_idx]['dt_creation_devis']

    train_cust = cust_first[cust_first['dt_creation_devis'] <= split_date]['numero_compte'].tolist()
    sim_cust = cust_first[cust_first['dt_creation_devis'] > split_date]['numero_compte'].tolist()

    df_train = df_quotes[df_quotes['numero_compte'].isin(train_cust)].copy()
    df_sim = df_quotes[df_quotes['numero_compte'].isin(sim_cust)].copy()

    print(f"Split: {len(train_cust)} train, {len(sim_cust)} sim customers")
    return {
        'train': df_train,
        'simulation': df_sim
    }