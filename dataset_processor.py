"""
combine_datasets.py

Combine WESAD, DREAMER, and ASCERTAIN into a single feature dataset.

Expected files:

- WESAD:
    ./data/processed/features_dataset.csv
    (your existing processed WESAD features)

- DREAMER:
    ./data/processed/dreamer_features_*.csv
    (created by your DREAMER processing pipeline)

- ASCERTAIN:
    ./data/ASCERTAIN_Features/
        Dt_ECGFeatures.mat
        Dt_GSRFeatures.mat
        Dt_EEGFeatures.mat
        Dt_EMOFeatures.mat
        Dt_SelfReports.mat
        Dt_Personality.mat
"""

import os
import glob
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import scipy.io


# ---------------------------------------------------------------------
# ASCERTAIN HELPERS
# ---------------------------------------------------------------------

def _unwrap_mat_struct(obj):
    """
    Helper to deal with MATLAB 1x1 struct arrays.

    Many .mat files (especially old ones) load into scipy as:
        array([[<np.void>]], dtype=object)
    where the inner element has .dtype.names.

    This unwraps that to the underlying structured element.
    """
    if isinstance(obj, np.ndarray):
        # Case 1: array of size (1,1) pointing to a struct
        if obj.size == 1:
            obj = obj.flat[0]
    return obj


def load_ascertain_feature_mats(base_path: str) -> Dict[str, np.ndarray]:
    """
    Load ASCERTAIN pre-extracted feature .mat files and normalize key names.

    Returns a dict with keys:
      - 'ECG', 'GSR', 'EEG', 'EMO'
      - 'Arousal', 'Valence', 'Engagement', 'Liking', 'Familiarity'
      - 'Personality', 'PermutationList'
    """
    feat_dir = os.path.join(base_path, "ASCERTAIN_Features")

    def _load_feat(path: str, key_candidates: List[str]) -> np.ndarray:
        mat = scipy.io.loadmat(path)
        visible_keys = [k for k in mat.keys() if not k.startswith("__")]
        print(f"[ASCERTAIN] {os.path.basename(path)} keys:", visible_keys)

        arr = None

        # 1) Exact match first
        for cand in key_candidates:
            if cand in mat:
                arr = mat[cand]
                print(f"[ASCERTAIN] Using exact key '{cand}' in {os.path.basename(path)}")
                break

        # 2) Fallback: prefix / substring (ECGFeatures -> ECGFeatures_58)
        if arr is None:
            for cand in key_candidates:
                for k in visible_keys:
                    if k.startswith(cand) or cand in k:
                        arr = mat[k]
                        print(f"[ASCERTAIN] Using key '{k}' (matched from '{cand}') in {os.path.basename(path)}")
                        break
                if arr is not None:
                    break

        if arr is None:
            raise ValueError(f"None of keys {key_candidates} found in {path} (visible: {visible_keys})")

        arr = np.array(arr)

        # --- CASE 1: MATLAB cell array -> object dtype, shape (58,1) or (1,58) etc ---
        if arr.dtype == object:
            # Flatten to 1D list of cells (one per subject)
            flat = arr.ravel()
            subj_arrays = []

            for i, cell in enumerate(flat):
                # Each cell itself may be nested; convert to numeric and squeeze
                subj = np.array(cell).squeeze()
                subj_arrays.append(subj)
                if i == 0:
                    print(f"[ASCERTAIN] Example inner shape for {os.path.basename(path)}:", subj.shape)

            # Now stack into (n_subjects, n_clips, feat_dim)
            arr_numeric = np.stack(subj_arrays, axis=0)
            print(f"[ASCERTAIN] Final stacked shape for {os.path.basename(path)}:", arr_numeric.shape)
            return arr_numeric

        # --- CASE 2: already numeric (e.g. 58 x 36 x D) ---
        arr = arr.squeeze()
        print(f"[ASCERTAIN] Numeric array shape for {os.path.basename(path)} after squeeze:", arr.shape)
        return arr



    # --- Load feature tensors ---
    ecg_path = os.path.join(feat_dir, "Dt_ECGFeatures.mat")
    gsr_path = os.path.join(feat_dir, "Dt_GSRFeatures.mat")
    eeg_path = os.path.join(feat_dir, "Dt_EEGFeatures.mat")
    emo_path = os.path.join(feat_dir, "Dt_EMOFeatures.mat")

    ECG = _load_feat(ecg_path, ["ECGFeatures"])
    GSR = _load_feat(gsr_path, ["GSRFeatures"])
    EEG = _load_feat(eeg_path, ["EEGFeatures"])
    EMO = _load_feat(emo_path, ["EMOFeatures"])

    print("[ASCERTAIN] ECG shape:", ECG.shape)
    print("[ASCERTAIN] GSR shape:", GSR.shape)
    print("[ASCERTAIN] EEG shape:", EEG.shape)
    print("[ASCERTAIN] EMO shape:", EMO.shape)

    # --- Self-reports ---
    sr_mat = scipy.io.loadmat(os.path.join(feat_dir, "Dt_SelfReports.mat"))
    print("[ASCERTAIN] SelfReports keys:", [k for k in sr_mat.keys() if not k.startswith("__")])

    if "Ratings" not in sr_mat:
        raise ValueError("Dt_SelfReports.mat does not contain 'Ratings'")

    R = np.array(sr_mat["Ratings"])
    # Shape: (5, NS, NV) — as seen from your print
    if R.ndim != 3 or R.shape[0] != 5:
        raise ValueError(f"Unexpected Ratings shape: {R.shape}, expected (5, NS, NV)")

    # axis 0: rating type, 1: subject, 2: video
    Arousal     = R[0].astype(float)   # NS x NV
    Valence     = R[1].astype(float)
    Engagement  = R[2].astype(float)
    Liking      = R[3].astype(float)
    Familiarity = R[4].astype(float)

    print("[ASCERTAIN] Arousal shape:", Arousal.shape)
    print("[ASCERTAIN] Valence shape:", Valence.shape)

    # --- Personality ---
    pers_mat = scipy.io.loadmat(os.path.join(feat_dir, "Dt_Personality.mat"))
    pers_keys = [k for k in pers_mat.keys() if not k.startswith("__")]
    print("[ASCERTAIN] Personality keys:", pers_keys)

    if "Personality" not in pers_mat:
        raise ValueError("Dt_Personality.mat does not contain 'Personality'")

    Personality = np.array(pers_mat["Personality"]).squeeze()  # NS x 5
    print("[ASCERTAIN] Personality shape:", Personality.shape)

    # --- Order (optional) ---
    order_mat = scipy.io.loadmat(os.path.join(feat_dir, "Dt_Order_Movie.mat"))
    order_keys = [k for k in order_mat.keys() if not k.startswith("__")]
    print("[ASCERTAIN] Order keys:", order_keys)

    PermutationList = None
    if "PermutationList" in order_mat:
        PermutationList = np.array(order_mat["PermutationList"]).squeeze()
        print("[ASCERTAIN] PermutationList shape:", PermutationList.shape)
    else:
        print("[ASCERTAIN] WARNING: 'PermutationList' not found in Dt_Order_Movie.mat")

    return {
        "ECG": ECG,
        "GSR": GSR,
        "EEG": EEG,
        "EMO": EMO,
        "Arousal": Arousal,
        "Valence": Valence,
        "Engagement": Engagement,
        "Liking": Liking,
        "Familiarity": Familiarity,
        "Personality": Personality,
        "PermutationList": PermutationList,
    }




def create_stress_label_from_valence_arousal(valence: float, arousal: float) -> int:
    """
    Binary stress label from A/V, consistent with your DREAMER rule.

    High arousal, low valence = Stress (1)
    Otherwise = Non-stress / neutral (0)
    """
    return int((arousal > 3) and (valence < 3))


def process_ascertain_features_to_df(base_path: str) -> pd.DataFrame:
    """
    Build a per-(subject, clip) feature DataFrame from ASCERTAIN precomputed features.

    One row per subject x video clip, with:
      - flattened ECG / GSR / EEG / EMO features
      - self-report ratings (A, V, E, L, F)
      - personality traits
      - binary stress_label derived from (A, V)
    """
    mats = load_ascertain_feature_mats(base_path)

    ECG = mats["ECG"]          # NS x NV x 32
    GSR = mats["GSR"]          # NS x NV x 31
    EEG = mats["EEG"]          # NS x NV x 88
    EMO = mats["EMO"]          # NS x NV x 72
    A   = mats["Arousal"]      # NS x NV
    V   = mats["Valence"]      # NS x NV
    E   = mats["Engagement"]   # NS x NV
    L   = mats["Liking"]       # NS x NV
    F   = mats["Familiarity"]  # NS x NV
    Personality = mats["Personality"]   # NS x 5  (Ex, Ag, Con, ES, O)

    print(f"[ASCERTAIN] ECG shape: {ECG.shape}")
    print(f"[ASCERTAIN] GSR shape: {GSR.shape}")
    print(f"[ASCERTAIN] EEG shape: {EEG.shape}")
    print(f"[ASCERTAIN] EMO shape: {EMO.shape}")
    print(f"[ASCERTAIN] Arousal shape: {A.shape}")
    print(f"[ASCERTAIN] Valence shape: {V.shape}")
    print(f"[ASCERTAIN] Engagement shape: {E.shape}")
    print(f"[ASCERTAIN] Liking shape: {L.shape}")
    print(f"[ASCERTAIN] Familiarity shape: {F.shape}")
    NS, NV, ecg_dim = ECG.shape
    _, _, gsr_dim = GSR.shape
    _, _, eeg_dim = EEG.shape
    _, _, emo_dim = EMO.shape

    print(f"[ASCERTAIN] NS={NS}, NV={NV}, dims: ECG={ecg_dim}, GSR={gsr_dim}, EEG={eeg_dim}, EMO={emo_dim}")

    rows = []

    for s in range(NS):
        ex, ag, co, es, op = Personality[s, :]

        for v_idx in range(NV):
            # self-report for that (subject, clip)
            a_val = float(A[s, v_idx])
            v_val = float(V[s, v_idx])
            e_val = float(E[s, v_idx])
            l_val = float(L[s, v_idx])
            f_val = float(F[s, v_idx])

            # Use the SAME rule as DREAMER/WESAD for binary stress:
            # high arousal & low valence (on the 1–5-ish scale) -> stress
            # This keeps the merged `stress_label` concept consistent.
            stress_label = create_stress_label_from_valence_arousal(
                valence=v_val,
                arousal=a_val,
            )

            row = {
                "dataset": "ASCERTAIN",
                "subject_id": f"ASC_{s+1}",
                "clip_id": v_idx + 1,
                "arousal": a_val,
                "valence": v_val,
                "engagement": e_val,
                "liking": l_val,
                "familiarity": f_val,
                "stress_label": stress_label,
                "Extraversion": ex,
                "Agreeableness": ag,
                "Conscientiousness": co,
                "EmotionalStability": es,
                "Openness": op,
            }

            # Flatten ECG features
            ecg_vec = ECG[s, v_idx, :].ravel()
            for k in range(ecg_dim):
                row[f"asc_ecg_{k+1:02d}"] = float(ecg_vec[k])

            # Flatten GSR features
            gsr_vec = GSR[s, v_idx, :].ravel()
            for k in range(gsr_dim):
                row[f"asc_gsr_{k+1:02d}"] = float(gsr_vec[k])

            # Flatten EEG features
            eeg_vec = EEG[s, v_idx, :].ravel()
            for k in range(eeg_dim):
                row[f"asc_eeg_{k+1:02d}"] = float(eeg_vec[k])

            # Flatten EMO features
            emo_vec = EMO[s, v_idx, :].ravel()
            for k in range(emo_dim):
                row[f"asc_emo_{k+1:02d}"] = float(emo_vec[k])

            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"[ASCERTAIN] Built dataframe with shape: {df.shape}")
    print("[ASCERTAIN] stress_label distribution:")
    print(df["stress_label"].value_counts(normalize=True).rename("proportion"))

    return df



# ---------------------------------------------------------------------
# WESAD + DREAMER LOADERS
# ---------------------------------------------------------------------

def load_wesad_df(processed_dir: str) -> pd.DataFrame:
    """
    Load WESAD processed windows from your existing CSV and
    create a unified stress_label.

    Assumes file: processed_dir/features_dataset.csv

    Assumes:
      - column 'label' with WESAD labels:
          1 = baseline, 2 = stress, 3 = amusement
    """
    path = os.path.join(processed_dir, "features_dataset.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"WESAD features not found at {path}")

    df = pd.read_csv(path)
    # Add dataset tag
    df["dataset"] = "WESAD"

    # Rename subject column if needed
    if "subject" not in df.columns:
        # adapt if you used a different name
        # e.g., 'subject_id' -> 'subject'
        for cand in ["subject_id", "subj_id", "S"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "subject"})
                break

    # Create unified binary stress_label
    if "stress_label" not in df.columns:
        if "label" not in df.columns:
            raise ValueError("WESAD dataframe has no 'label' column for stress mapping.")
        df["stress_label"] = df["label"].apply(lambda x: 1 if x == 2 else 0)

    print(f"[WESAD] Loaded with shape: {df.shape}")
    print(df[["stress_label"]].value_counts(normalize=True, dropna=False))
    return df


def load_dreamer_df(processed_dir: str) -> pd.DataFrame:
    """
    Load all DREAMER window-level CSVs and merge.

    Assumes files: processed_dir/dreamer_features_*.csv

    Must contain:
      - 'stress_label'
      - some subject column (we normalize to 'subject')
    """
    pattern = os.path.join(processed_dir, "dreamer_features_*.csv")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"No DREAMER feature files found with pattern {pattern}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["dataset"] = "DREAMER"

        # Normalize subject column
        if "subject" not in df.columns:
            for cand in ["subject_id", "subj_id", "S"]:
                if cand in df.columns:
                    df = df.rename(columns={cand: "subject"})
                    break

        if "subject" not in df.columns:
            # if we truly have no subject info, fallback to a dummy
            df["subject"] = 1

        # Ensure stress_label exists (your DREAMER pipeline already sets this)
        if "stress_label" not in df.columns:
            if {"valence", "arousal"}.issubset(df.columns):
                df["stress_label"] = df.apply(
                    lambda r: create_stress_label_from_valence_arousal(
                        r["valence"], r["arousal"]
                    ),
                    axis=1,
                )
            else:
                raise ValueError(f"DREAMER file {f} has no 'stress_label' or A/V cols.")

        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    print(f"[DREAMER] Loaded {len(files)} files, merged shape: {merged.shape}")
    print(merged[["stress_label"]].value_counts(normalize=True, dropna=False))
    return merged


# ---------------------------------------------------------------------
# MERGING
# ---------------------------------------------------------------------

def merge_all_datasets(
    wesad_base: str,
    dreamer_base: str,
    ascertain_base: str,
    n_windows_per_ascertain_clip: int = 5,
    save_path: str = "./data"
):
    """
    Build a unified dataset containing WESAD, DREAMER, ASCERTAIN.

    Args:
        wesad_dir: directory where WESAD CSV lives
        dreamer_dir: directory where DREAMER CSVs live
        ascertain_base: directory that contains ASCERTAIN_Features/
        output_path: final CSV output path

    Returns:
        merged_df, column_counts
    """

    def expand_ascertain_to_windows(asc_df: pd.DataFrame,
                                n_windows_per_clip: int = 5) -> pd.DataFrame:
        """
        Option 2: approximate sliding windows for ASCERTAIN by replicating each
        clip-level row into `n_windows_per_clip` pseudo-windows.

        - Keeps all ASCERTAIN features as-is for every pseudo-window.
        - Adds a `window_id` column (0..n_windows_per_clip-1).
        - Ensures there is a `subject` column compatible with WESAD/DREAMER.
        """
        asc_df = asc_df.copy()

        # Make sure dataset column exists
        if "dataset" not in asc_df.columns:
            asc_df["dataset"] = "ASCERTAIN"
        else:
            asc_df["dataset"] = asc_df["dataset"].fillna("ASCERTAIN")

        # Ensure a unified `subject` column (WESAD & DREAMER already have `subject`)
        if "subject" not in asc_df.columns:
            if "subject_id" in asc_df.columns:
                asc_df["subject"] = asc_df["subject_id"].astype(str)
            else:
                # Fallback: build subject from index if needed
                asc_df["subject"] = asc_df.index.astype(str)
        else:
            asc_df["subject"] = asc_df["subject"].astype(str)

        # Now replicate rows
        repeated_rows = []
        for _, row in asc_df.iterrows():
            for w in range(n_windows_per_clip):
                new_row = row.copy()
                new_row["window_id"] = w   # pseudo-window index
                repeated_rows.append(new_row)

        asc_windows_df = pd.DataFrame(repeated_rows)
        # For compatibility with WESAD/DREAMER, ensure `window_id` exists there too
        return asc_windows_df

    wesad_df = load_wesad_df(wesad_base)
    dreamer_df = load_dreamer_df(dreamer_base)
    ascertain_df = process_ascertain_features_to_df(ascertain_base)

    # ---- 2. Normalize subject IDs for WESAD & DREAMER ----
    if "subject" in wesad_df.columns:
        wesad_df["subject"] = wesad_df["subject"].astype(str)
    if "subject" in dreamer_df.columns:
        dreamer_df["subject"] = dreamer_df["subject"].astype(str)

    # Ensure they also have a window_id column (for consistency)
    if "window_id" not in wesad_df.columns:
        # if you already have a window index column, use that instead
        wesad_df["window_id"] = wesad_df.groupby("subject").cumcount()
    if "window_id" not in dreamer_df.columns:
        dreamer_df["window_id"] = dreamer_df.groupby("subject").cumcount()

    # ---- 3. Expand ASCERTAIN to pseudo-windows (Option 2) ----
    asc_windows_df = expand_ascertain_to_windows(
        ascertain_df,
        n_windows_per_clip=n_windows_per_ascertain_clip
    )

    # ---- 4. Concatenate all three datasets ----
    merged_df = pd.concat(
        [wesad_df, dreamer_df, asc_windows_df],
        ignore_index=True,
        sort=False
    )

    # ---- 5. Basic stats ----
    stats = {}
    stats["total_rows"] = len(merged_df)
    stats["per_dataset_counts"] = merged_df["dataset"].value_counts()
    stats["stress_distribution_overall"] = (
        merged_df["stress_label"].value_counts(normalize=True)
        .rename("proportion")
    )
    stats["stress_distribution_by_dataset"] = (
        merged_df
        .groupby("dataset")["stress_label"]
        .value_counts(normalize=True)
        .rename("proportion")
        .reset_index()
    )

    print(f"Merged shape: {merged_df.shape}")
    print("Dataset counts:")
    print(stats["per_dataset_counts"])
    print("Stress label distribution (overall):")
    print(stats["stress_distribution_overall"])

    # ---- 6. Save outputs ----
    os.makedirs(save_path, exist_ok=True)
    merged_path = os.path.join(save_path, "stress_merged_all_pseudowindows.csv")
    merged_df.to_csv(merged_path, index=False)
    print(f"[SAVE] Merged dataset with pseudo-windows saved to: {merged_path}")

    # Also save a clean copy of the original clip-level ASCERTAIN if you want
    asc_clip_path = os.path.join(save_path, "stress_ascertain_clip_level.csv")
    ascertain_df.to_csv(asc_clip_path, index=False)
    print(f"[SAVE] ASCERTAIN clip-level dataset saved to: {asc_clip_path}")

    return merged_df, stats


if __name__ == "__main__":
    wesad_dir = "./data/processed"
    dreamer_dir = "./data/processed"
    ascertain_dir = "./data"

    merged_df, stats = merge_all_datasets(
        wesad_base=wesad_dir,
        dreamer_base=dreamer_dir,
        ascertain_base=ascertain_dir,
        n_windows_per_ascertain_clip=5,
        save_path="./data",
    )
