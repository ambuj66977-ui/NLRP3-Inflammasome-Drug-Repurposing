# Final Project Report: ML-Based FDA Drug Repurposing for NLRP3 Inflammasome

## 1. Executive Summary
This project outlines the computational framework to repurpose existing, FDA-approved drugs for use strictly as **modulators** against the **NLRP3 Inflammasome**. Because traditional drug discovery takes decades and billions of dollars, we computationally simulated the binding inhibition (`pIC50`) of historical FDA-approved molecules using Machine Learning (ML), drastically reducing time-to-market.

Instead of predicting generic "drug-disease" vectors, this repository is built precisely on the physical realities of **Molecular Bindings and Topologies**. 

---

## 2. Defining The Biological & Chemical Problem
The NLRP3 inflammasome requires ligands to traverse the intracellular membrane and physically interact with its catalytic binding pockets (e.g. NACHT domain). 

We solved this requirement by extracting:
- **Physicochemical Properties**: Lipinski's Rule of 5 (Molecular Weight, AlogP) determining cell permeability and lipophilicity.
- **Morgan Fingerprints**: Utilizing RDKit to construct 2D structural graphs mapping the distinct atomic branches, rings, and Hydrogen-Bond Acceptors/Donors to predict "locks and keys".

By doing so, we transitioned from basic text-based parameters into a complex Structure-Based Machine Learning application. 

---

## 3. Data Methodologies & Mitigation of Data Leakage
The precursory version of this pipeline inadvertently incorporated "Ligand Efficiency" (LE, LLE, BEI) metrics as predictive properties, artificially creating a 0.94 $R^2$ validation score. Since LE is a direct mathematical derivative of affinity `$pIC50 / Heavy\_Atom\_Count$`, this constituted massive target leakage.

The refined pipeline structurally overhauled the variables:
1. **Cleaning Process**: Dropped all leaky identifiers. Extracted canonical `SMILES` to dynamically build 2048-bit Fingerprint signatures for all 837 active/inactive ChEMBL/NLRP3 datasets.
2. **Model Training**: Standard-Scaled robust descriptors with random selection partitioning, feeding directly into a Random Forest Regressor tree depth.
3. **Realistic Output**: The new architecture accurately reported a **Test $R^2$ of 0.738** with an RMSE of 0.681 `pIC50`. This translates to genuine learning of molecular geometry instead of math reversal.

---

## 4. Virtual Screening Architecture (The Repurposing Engine)
The fundamental requirement of the project was screening existing compounds. The custom pipeline (`03_repurpose_fda.py`) integrated with external databases (e.g. public ClinTox endpoints/FDA SMILES tables) extracting over **1,386 verifiable, safe FDA drugs**.

For each drug, the python loop:
- Inferred Lipinski RO5 violations computationally.
- Placed the canonical SMILES string into the `RDKit` mathematical transformer. 
- Sent the 2051 standardized feature weights into the finalized `nlrp3_rf_model.pkl` Random Forest model.

---

## 5. Candidate Validation & Results Output
The automated engine successfully generated and ranked theoretical candidates purely computationally without human bias. The ranking generated a list of **Top 100 potential hits**. The leading matches projected heavily favorable predicted `pIC50` readings nearing the **7.45 - 7.39** bounds.

> **Note on Industry Next Steps:**  
> The computational ranking of the SMILES represents the first phase of Drug Repurposing. The generated `.csv` hits (located in `/results`) must now be submitted to Structure-Based pipelines like AutoDock Vina, running physics-driven Molecular Dynamics simulations against the 3D target coordinates (PDB IDs: 7ALV, 6NPY).
