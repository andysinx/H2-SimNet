import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
import torch

# --- Moving average filter ---
def apply_moving_average(df, feature_cols, window=5):
    df_filtered = df.copy()
    for col in feature_cols:
        df_filtered[col] = df_filtered[col].rolling(window=window, min_periods=1).mean()
    return df_filtered

# --- Sliding windows ---
def create_windows(df, feature_cols, window_size=5, step=1):
    data = df[feature_cols].values
    return np.array([data[i:i+window_size] for i in range(0, len(data)-window_size+1, step)], dtype=np.float32)

# --- Positive windows from anomalies ---
def create_positive_windows(df, feature_cols, window_size=50, step=1, label_col="label_anomaly"):
    anomaly_start = df[df[label_col]==1].index[0]
    df_anom = df.loc[anomaly_start:]
    data = df_anom[feature_cols].values
    return np.array([data[i:i+window_size] for i in range(0, len(data)-window_size+1, step)], dtype=np.float32)

# --- Load normal data ---
def load_normal_data(feature_cols, normal_paths, window_size=50, step=1, derivatives=False, filt_win_size=None):
    windows_list = []
    for path in normal_paths:
        df = pd.read_parquet(path)
        if isinstance(filt_win_size, int) and filt_win_size>1:
            df = apply_moving_average(df, feature_cols, window=filt_win_size)
        w = create_windows(df, feature_cols, window_size, step)
        if derivatives:
            delta_t = 1.0
            deriv = np.diff(w, axis=1, prepend=w[:, :1, :]) / delta_t
            w = np.concatenate([w, deriv], axis=2)
        windows_list.append(w)
    windows = np.concatenate(windows_list, axis=0)
    y = np.zeros(len(windows), dtype=int)
    return windows, y

# --- Load anomalous data ---
def load_anomalous_data(anomalous_file, feature_cols, window_size=50, step=1, label_col="label_anomaly",
                        position='assestment_transient', derivatives=False):
    topologies = ["anom_gestcent", "anom_gestdown", "anom_gestup"]
    windows_list = []
    for topo in topologies:
        full_path = Path(anomalous_file).parent.parent / topo / position / Path(anomalous_file).name
        if full_path.exists():
            df = pd.read_parquet(full_path)
            w = create_positive_windows(df, feature_cols, window_size, step, label_col)
            if derivatives:
                delta_t = 1.0
                deriv = np.diff(w, axis=1, prepend=w[:, :1, :]) / delta_t
                w = np.concatenate([w, deriv], axis=2)
            windows_list.append(w)
    if len(windows_list)==0:
        num_features = len(feature_cols)*(2 if derivatives else 1)
        return np.empty((0, window_size, num_features), dtype=np.float32), np.empty(0, dtype=int)
    windows = np.concatenate(windows_list, axis=0)
    y = np.ones(len(windows), dtype=int)
    return windows, y

# --- Main unsupervised dataset builder ---
def create_unsupervised_dataset(anomalous_file, window_size=50, step=1, derivatives=False,
                                position='assestment_transient', filt_win_size=None,
                                k_fold=None, batch_size=64, test_split=0.2, max_anom_ratio=0.05):

    PS_cols = [f"PS{i}" for i in range(1, 13)]
    MFS_cols = [f"MFS{i}" for i in range(1, 5)]
    feature_cols = PS_cols + MFS_cols

    # Normal data paths
    normal_paths = [
        "../normality-scenarios/parquet/normality_gest_down.parquet",
        "../normality-scenarios/parquet/normality_gest_center.parquet",
        "../normality-scenarios/parquet/normality_gest_up.parquet"
    ]
    clean_paths = [
        "../H2-SimNet-clean/normality-scenarios/parquet/normality_gest_down.parquet",
        "../H2-SimNet-clean/normality-scenarios/parquet/normality_gest_center.parquet",
        "../H2-SimNet-clean/normality-scenarios/parquet/normality_gest_up.parquet"
    ]

    # Shuffle paths before loading
    all_normal_paths = normal_paths + clean_paths
    np.random.shuffle(all_normal_paths)

    # --- Load all normal data ---
    normal_windows, y_normal = load_normal_data(feature_cols, all_normal_paths, window_size, step, derivatives, filt_win_size)

    # --- Split train/test normali ---
    X_trainval, X_test_norm, _, _ = train_test_split(
        normal_windows, y_normal, test_size=test_split, random_state=42
    )

    # --- Load anomalous data for test ---
    X_test_anom, y_test_anom = load_anomalous_data(anomalous_file, feature_cols, window_size, step, position=position, derivatives=derivatives)

    # --- Limit anomalie nel test set ---
    num_norm = len(X_test_norm)
    max_anom = int(num_norm * max_anom_ratio / (1 - max_anom_ratio))
    if len(X_test_anom) > max_anom:
        X_test_anom = X_test_anom[:max_anom]
        y_test_anom = y_test_anom[:max_anom]

    # --- Final test set ---
    X_test = np.concatenate([X_test_norm, X_test_anom], axis=0)
    y_test = np.concatenate([np.zeros(len(X_test_norm), dtype=int), y_test_anom], axis=0)

    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                                  torch.tensor(y_test, dtype=torch.float32))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- K-Fold for training and validation with anomalies ---
    folds = []
    if k_fold is not None:
        kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
        X_val_anom, y_val_anom = load_anomalous_data(anomalous_file, feature_cols, window_size, step, position=position, derivatives=derivatives)

        for train_idx, val_idx in kf.split(X_trainval):
            X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
            y_train = np.zeros(len(train_idx))  # only normal

            # Add anomalies to validation (max 5%)
            max_anom_fold = int(len(X_val) * max_anom_ratio)
            if len(X_val_anom) > max_anom_fold:
                X_val_anom_fold = X_val_anom[:max_anom_fold]
                y_val_anom_fold = y_val_anom[:max_anom_fold]
            else:
                X_val_anom_fold = X_val_anom
                y_val_anom_fold = y_val_anom

            X_val_fold = np.concatenate([X_val, X_val_anom_fold], axis=0)
            y_val_fold = np.concatenate([np.zeros(len(X_val)), y_val_anom_fold], axis=0)

            # Shuffle validation set
            perm = np.random.permutation(len(X_val_fold))
            X_val_fold, y_val_fold = X_val_fold[perm], y_val_fold[perm]

            # --- Torch loaders ---
            train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                           torch.tensor(y_train, dtype=torch.float32))
            val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val_fold, dtype=torch.float32),
                                                         torch.tensor(y_val_fold, dtype=torch.float32))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            folds.append((train_loader, val_loader))

    return folds, test_loader
