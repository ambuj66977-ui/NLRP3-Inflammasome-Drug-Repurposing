import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem

def prepare_data(input_csv: str, output_csv: str):
    print(f"Loading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # 1. Drop Leaky Features
    leakage_cols = ["Ligand Efficiency LE", "Ligand Efficiency LLE", "Ligand Efficiency BEI"]
    cols_to_drop = [c for c in leakage_cols if c in df.columns]
    if cols_to_drop:
        print(f"Dropping leakage columns: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)
        
    # We might have `Action_Encoded` missing from the old dataset but have `pIC50`. 
    # Also drop any Unnamed index columns
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    # Make sure pIC50 exists
    if "pIC50" not in df.columns:
        raise ValueError("Target variable 'pIC50' is missing from the dataset!")

    print(f"Dataset contains {len(df)} records. Starting featurization...")

    # 2. Extract Morgan Fingerprints (structural topology)
    # We need the "Smiles" column
    if "Smiles" not in df.columns:
        # Check standard casing
        if "SMILES" in df.columns:
            df.rename(columns={"SMILES": "Smiles"}, inplace=True)
        else:
            raise ValueError("Dataset must contain a 'Smiles' column to generate fingerprints!")

    failed_idxs = []
    fps_list = []
    
    for idx, smi in enumerate(df["Smiles"]):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                failed_idxs.append(idx)
                fps_list.append(np.zeros((2048,)))
                continue
            
            # Generate Morgan fingerprint (Radius 2, 2048 bits)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fp_arr = np.zeros((1,))
            from rdkit.DataStructs import ConvertToNumpyArray
            ConvertToNumpyArray(fp, fp_arr)
            fps_list.append(fp_arr)
            
        except Exception as e:
            failed_idxs.append(idx)
            fps_list.append(np.zeros((2048,)))
            
    print(f"Successfully processed {len(df) - len(failed_idxs)} SMILES. Failed: {len(failed_idxs)}")
    
    # Create DataFrame for fingerprints
    fp_cols = [f"FP_{i}" for i in range(2048)]
    fp_df = pd.DataFrame(fps_list, columns=fp_cols, index=df.index)
    
    # Merge Features
    final_df = pd.concat([df, fp_df], axis=1)
    
    # Drop rows where SMILES failed parsing entirely
    if failed_idxs:
        final_df.drop(index=failed_idxs, inplace=True)
    
    # Save Output
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    print(f"Saved cleaned and featurized data to {output_csv}. Total features: {len(final_df.columns)}")

if __name__ == "__main__":
    base_dir = "/Users/ambuj/Desktop/drugrepurposing "
    in_file = os.path.join(base_dir, "data", "nlrp3_ml_ready_dataset.csv")
    out_file = os.path.join(base_dir, "data", "nlrp3_featurized_clean.csv")
    prepare_data(in_file, out_file)
