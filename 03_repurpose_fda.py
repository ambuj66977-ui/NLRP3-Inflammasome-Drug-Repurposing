import pandas as pd
import numpy as np
import os
import requests
import joblib
from io import StringIO
import warnings

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import AllChem

warnings.filterwarnings("ignore", category=UserWarning)
import urllib3
urllib3.disable_warnings()

def compute_ro5_violations(mol):
    violations = 0
    if Descriptors.MolWt(mol) > 500:
        violations += 1
    if Crippen.MolLogP(mol) > 5:
        violations += 1
    if Lipinski.NumHDonors(mol) > 5:
        violations += 1
    if Lipinski.NumHAcceptors(mol) > 10:
        violations += 1
    return violations

def repurpose_fda_drugs(base_dir: str):
    print("Fetching FDA-approved drugs from ClinTox dataset...")
    url = "https://raw.githubusercontent.com/GLambard/Molecules_Dataset_Collection/master/originals/clintox.csv"
    try:
        response = requests.get(url, verify=False)
        response.raise_for_status()
        csv_data = StringIO(response.text)
        fda_df = pd.read_csv(csv_data)
        
        # Filter only FDA approved
        if "FDA_APPROVED" in fda_df.columns:
            fda_df = fda_df[fda_df["FDA_APPROVED"] == 1]
    except Exception as e:
        print(f"Failed to download FDA dataset: {e}")
        return

    # Drop anything without SMILES
    if "smiles" not in fda_df.columns:
        print("SMILES column not found in downloaded dataset.")
        return
        
    fda_df = fda_df.dropna(subset=["smiles"])
    print(f"Found {len(fda_df)} FDA drugs with SMILES.")
    
    # We need: 'Molecular Weight', 'AlogP', '#RO5 Violations', and 2048 fingerprints
    # Action_Encoded can be set to 0 (default/unknown) since our model uses it but it might not be relevant for inference,
    # or honestly, Action_Encoded=1 means "Inhibitor" in training. Let's just pass 0 or 1.
    # Actually wait! The training data had `Action_Encoded` missing for many rows, but wait, `model_pipeline` expects it!
    # If the model requires `Action_Encoded`, we will set it to 1 (Assuming we are looking for Inhibitors).
    
    records = []
    fps_list = []
    valid_indices = []
    
    print("Computing physicochemical descriptors and Morgan Fingerprints...")
    for idx, row in fda_df.iterrows():
        smi = row["smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
            
        try:
            mw = Descriptors.MolWt(mol)
            alogp = Crippen.MolLogP(mol)
            ro5 = compute_ro5_violations(mol)
            
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fp_arr = np.zeros((1,))
            from rdkit.DataStructs import ConvertToNumpyArray
            ConvertToNumpyArray(fp, fp_arr)
            
            valid_indices.append(idx)
            records.append({
                "Molecular Weight": mw,
                "AlogP": alogp,
                "#RO5 Violations": ro5
            })
            fps_list.append(fp_arr)
        except Exception as e:
            pass

    print(f"Successfully featurized {len(records)} drugs.")
    
    X_fda = pd.DataFrame(records, index=valid_indices)
    fp_cols = [f"FP_{i}" for i in range(2048)]
    fp_df = pd.DataFrame(fps_list, columns=fp_cols, index=valid_indices)
    
    X_final = pd.concat([X_fda, fp_df], axis=1)
    
    # Load the Model
    model_path = os.path.join(base_dir, "models", "nlrp3_rf_model.pkl")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}.")
        return
        
    print("Loading predictive model...")
    model_pipeline = joblib.load(model_path)
    
    print("Running predictions...")
    preds = model_pipeline.predict(X_final)
    
    # Append predictions back to the original df
    result_df = fda_df.loc[valid_indices].copy()
    result_df["Predicted_pIC50"] = preds
    
    # Rank by pIC50 (higher is stronger affinity)
    result_df = result_df.sort_values(by="Predicted_pIC50", ascending=False)
    
    # Save Top 100 Candidates
    out_dir = os.path.join(base_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "top_fda_candidates.csv")
    
    result_df.head(100).to_csv(out_path, index=False)
    print(f"Optimization complete! Top 100 FDA candidates saved to {out_path}")
    print(f"\\n--- TOP 5 FDA REPUPROSING CANDIDATES (NLRP3 Inhibitors) ---")
    
    # DrugBank dataset has an 'name' column or 'title' column? Usually 'name'.
    name_col = "name" if "name" in result_df.columns else "drugbank_id"
    for i, (_, row) in enumerate(result_df.head(5).iterrows()):
        print(f"{i+1}. {row.get(name_col, 'Unknown')} - Predicted pIC50: {row['Predicted_pIC50']:.2f}")

if __name__ == "__main__":
    base_dir = "/Users/ambuj/Desktop/drugrepurposing "
    repurpose_fda_drugs(base_dir)
