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
    anomaly_idx = df[df[label_col]==1].index
    if len(anomaly_idx) == 0:
        return np.empty((0, window_size, len(feature_cols)), dtype=np.float32)

    df_anom = df.loc[anomaly_idx[0]:]
    data = df_anom[feature_cols].values

    # se la sequenza è più corta di window_size, pad con l'ultimo valore
    if len(data) < window_size:
        pad = np.tile(data[-1], (window_size - len(data), 1))
        data = np.vstack([data, pad])
        return np.expand_dims(data, axis=0)

    windows = np.array([data[i:i+window_size] for i in range(0, len(data)-window_size+1, step)], dtype=np.float32)
    return windows

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
    if len(windows_list) == 0:
        num_features = len(feature_cols)*(2 if derivatives else 1)
        return np.empty((0, window_size, num_features), dtype=np.float32), np.empty(0, dtype=int)
    windows = np.concatenate(windows_list, axis=0)
    y = np.ones(len(windows), dtype=int)
    return windows, y

# --- Main unsupervised dataset builder ---
def create_unsupervised_dataset(anomalous_file, position, window_size=50, step=1, derivatives=False,
                                 k_fold=None, batch_size=64, test_split=0.2, val_anom_ratio=0.05):
    PS_cols = [f"PS{i}" for i in range(1, 13)]
    MFS_cols = [f"MFS{i}" for i in range(1, 5)]
    feature_cols = PS_cols + MFS_cols

    # --- Load normal data ---
    normal_paths = [
        "../normality-scenarios/parquet/normality_gest_down.parquet",
        "../normality-scenarios/parquet/normality_gest_center.parquet",
        "../normality-scenarios/parquet/normality_gest_up.parquet"
    ]

    # --- Load normal data ---
    clean_paths = [
        "../H2-SimNet-clean/normality-scenarios/parquet/normality_gest_down.parquet",
        "../H2-SimNet-clean/normality-scenarios/parquet/normality_gest_center.parquet",
        "../H2-SimNet-clean/normality-scenarios/parquet/normality_gest_up.parquet"
    ]
    all_normal_paths = normal_paths + clean_paths
    np.random.shuffle(all_normal_paths)
    normal_windows, y_normal = load_normal_data(feature_cols, all_normal_paths, window_size, step, derivatives)

    # --- Split train+validation normal vs test normal ---
    X_trainval_norm, X_test_norm, _, _ = train_test_split(normal_windows, y_normal, test_size=test_split, random_state=42)

    # --- Load all anomalous data ---
    X_anom_all, y_anom_all = load_anomalous_data(anomalous_file, feature_cols, window_size, step, position=position, derivatives=derivatives)
    num_val_anom = max(1, int(len(X_anom_all) * val_anom_ratio))
    perm = np.random.permutation(len(X_anom_all))
    X_val_anom = X_anom_all[perm[:num_val_anom]]
    y_val_anom = y_anom_all[perm[:num_val_anom]]
    X_test_anom = X_anom_all[perm[num_val_anom:]]
    y_test_anom = y_anom_all[perm[num_val_anom:]]

    # --- Final test set (normali + anomalie rimanenti) ---
    X_test = np.concatenate([X_test_norm, X_test_anom], axis=0)
    y_test = np.concatenate([np.zeros(len(X_test_norm), dtype=int), y_test_anom], axis=0)
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                                  torch.tensor(y_test, dtype=torch.float32))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- K-Fold training+validation ---
    folds = []
    if k_fold is not None:
        kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X_trainval_norm):
            X_train, X_val = X_trainval_norm[train_idx], X_trainval_norm[val_idx]
            y_train = np.zeros(len(X_train), dtype=int)  # solo normali

            # Add small anomalous fraction only to validation
            X_val_fold = np.concatenate([X_val, X_val_anom], axis=0)
            y_val_fold = np.concatenate([np.zeros(len(X_val), dtype=int), y_val_anom], axis=0)

            # Shuffle validation set
            perm_val = np.random.permutation(len(X_val_fold))
            X_val_fold, y_val_fold = X_val_fold[perm_val], y_val_fold[perm_val]

            # --- Torch loaders ---
            train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                           torch.tensor(y_train, dtype=torch.float32))
            val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val_fold, dtype=torch.float32),
                                                         torch.tensor(y_val_fold, dtype=torch.float32))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            folds.append((train_loader, val_loader))

    return folds, test_loader



