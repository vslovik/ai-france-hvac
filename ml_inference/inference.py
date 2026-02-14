import pandas as pd

from ml_features.features import create_features
from ml_simulation.util import HiddenPrints


def safe_predict(customer_id, quotes_df, model, feature_names):
    """
    Safely predict probability even if features are missing.
    """
    with HiddenPrints():
        features_df = create_features(quotes_df)

    cust_features = features_df[features_df['numero_compte'] == customer_id]

    if len(cust_features) == 0:
        cust_features = pd.DataFrame({'numero_compte': [customer_id]})

    X_dict = {}
    for feat in feature_names:
        if feat in cust_features.columns:
            X_dict[feat] = cust_features[feat].iloc[0] if len(cust_features) > 0 else 0
        else:
            X_dict[feat] = 0

    X_cust = pd.DataFrame([X_dict])

    # try:
    prob = model.predict_proba(X_cust[feature_names])[:, 1][0]
    # except:
    #     prob = 0.5

    return prob