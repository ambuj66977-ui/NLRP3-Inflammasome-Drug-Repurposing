import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def train_and_save_model(input_csv: str, model_out: str):
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # 1. Define Target and Features
    target = "pIC50"
    
    # Drop identifiers and textual data if any
    cols_to_drop = [target, "Smiles", "Compound Name", "ChEMBL_ID"]
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y = df[target]

    # Morgan fingerprints are already binary/numeric, identify basic descriptors
    standard_features = ["Molecular Weight", "AlogP", "#RO5 Violations"]
    standard_features = [f for f in standard_features if f in X.columns]
    
    fp_features = [f for f in X.columns if f.startswith("FP_")]
    action_features = [f for f in X.columns if "Action" in f]
    
    print(f"Features: {len(standard_features)} Physicochemical, {len(fp_features)} Fingerprints")
    
    # 2. Build Pipeline
    # We only standard scale the physicochemical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), standard_features),
            # Pass through fingerprints and binary encoded features
            ("bin", "passthrough", fp_features + action_features)
        ],
        remainder='drop'  # Drop anything else not matched
    )
    
    # Random Forest Regressor
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", rf_model)
    ])
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training shapes -> X: {X_train.shape}, Test shapes -> X: {X_test.shape}")
    
    # 4. Train Model
    print("Training Random Forest Regressor...")
    model_pipeline.fit(X_train, y_train)
    
    # 5. Evaluate
    y_pred_test = model_pipeline.predict(X_test)
    y_pred_train = model_pipeline.predict(X_train)
    
    r2_test = r2_score(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print("\n--- Model Evaluation ---")
    print(f"Train R2:  {r2_train:.3f}")
    print(f"Test R2:   {r2_test:.3f}   <-- This is realistic for biology!")
    print(f"Test RMSE: {rmse_test:.3f} pIC50 units")
    
    # 6. Save Pipeline
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model_pipeline, model_out)
    print(f"\nModel strictly saved to {model_out}")

if __name__ == "__main__":
    base_dir = "/Users/ambuj/Desktop/drugrepurposing "
    in_file = os.path.join(base_dir, "data", "nlrp3_featurized_clean.csv")
    model_file = os.path.join(base_dir, "models", "nlrp3_rf_model.pkl")
    
    train_and_save_model(in_file, model_file)
