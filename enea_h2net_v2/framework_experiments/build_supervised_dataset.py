import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch

# --- Moving average filter ---
def apply_moving_average(df, feature_cols, window=5):
    df_filtered = df.copy()
    for col in feature_cols:
        df_filtered[col] = df_filtered[col].rolling(window=window, min_periods=1, center=False).mean()
    return df_filtered

# --- Function to create sliding windows ---
def create_windows(df, feature_cols, window_size=5, step=1):
    data = df[feature_cols].values
    return np.array([data[i:i+window_size] for i in range(0, len(data)-window_size+1, step)], dtype=np.float32)

# --- Function to create positive windows from anomalies ---
def create_positive_windows(df, feature_cols, window_size=50, 
                            step=1, label_col="label_anomaly"):
    anomaly_start = df[df[label_col] == 1].index[0]
    df_anomaly = df.loc[anomaly_start:]
    data = df_anomaly[feature_cols].values
    return np.array([data[i:i+window_size] for i in range(0, len(data)-window_size+1, step)], dtype=np.float32)

# --- Function to load normal data (both clean and noisy) ---
def load_normal_data(feature_cols, window_size=50, 
                     step=1, derivatives=False, filt_win_size=False):
    normal_paths_clean = []
    normal_paths_noisy = [
        "../normality-scenarios/parquet/normality_gest_down.parquet",
        "../normality-scenarios/parquet/normality_gest_center.parquet",
        "../normality-scenarios/parquet/normality_gest_up.parquet"
    ]
    
    windows_list = []
    for path in normal_paths_clean + normal_paths_noisy:
        df = pd.read_parquet(path)

        if isinstance(filt_win_size, int) and filt_win_size > 1:
            df = apply_moving_average(df, feature_cols, window=filt_win_size)

        w = create_windows(df, feature_cols, window_size, step)  # shape (num_windows, window_size, num_features)
        
        if derivatives:
            delta_t = 1.0
            # derivata finita lungo i timesteps (axis=1)
            deriv = np.diff(w, axis=1, prepend=w[:, :1, :]) / delta_t  # prepend primo timestep per mantenere shape
            # concatena le derivata alle feature originali
            w = np.concatenate([w, deriv], axis=2)  # nuova shape (num_windows, window_size, num_features*2)
        
        windows_list.append(w)
    
    windows = np.concatenate(windows_list, axis=0)
    y = np.zeros(len(windows), dtype=int)
    
    # Shuffle
    indices = np.arange(len(windows))
    np.random.shuffle(indices)
    return windows[indices], y[indices]

# --- Function to load anomalous data for all topologies ---
def load_anomalous_data(anomalous_file, feature_cols, window_size=50, step=1,
                        label_col="label_anomaly", position='assestment_transient',
                        derivatives=False):
    topologies = ["anom_gestcent", "anom_gestdown", "anom_gestup"]
    windows_list = []
    
    for topo in topologies:
        # Qui inseriamo la sottocartella assestment_transient
        full_path = Path(anomalous_file).parent.parent / topo / position / Path(anomalous_file).name
        if full_path.exists():
            df = pd.read_parquet(full_path)
            w = create_positive_windows(df, feature_cols, window_size, step, label_col)  # shape (num_windows, window_size, num_features)
            
            if derivatives:
                delta_t = 1.0
                deriv = np.diff(w, axis=1, prepend=w[:, :1, :]) / delta_t
                w = np.concatenate([w, deriv], axis=2)  # nuova shape (num_windows, window_size, num_features*2)
            
            windows_list.append(w)
        else:
            print(f"File not found: {full_path}")
    
    if len(windows_list) == 0:
        num_features = len(feature_cols) * (2 if derivatives else 1)
        return np.empty((0, window_size, num_features), dtype=np.float32), np.empty(0, dtype=int)
    
    windows = np.concatenate(windows_list, axis=0)
    y = np.ones(len(windows), dtype=int)
    return windows, y

# --- Main function to create balanced dataset ---
def create_balanced_dataset(anomalous_file, window_size=50, 
                                      step=1, derivatives=False, 
                                      position='assestment_transient', filt_win_size=None,
                                      k_fold=None, batch_size=64, test_split=0.2):

    PS_cols = [f"PS{i}" for i in range(1, 13)]
    MFS_cols = [f"MFS{i}" for i in range(1, 5)]
    feature_cols = PS_cols + MFS_cols

    # --- Carica dati ---
    normal_windows, y_normal = load_normal_data(feature_cols, window_size, step, derivatives, filt_win_size=filt_win_size)
    positive_windows, y_positive = load_anomalous_data(anomalous_file, feature_cols, window_size, step, position=position, derivatives=derivatives)

    # Subsample normal windows per bilanciare
    num_pos = len(positive_windows)
    if len(normal_windows) > num_pos:
        normal_windows = normal_windows[:num_pos]
        y_normal = y_normal[:num_pos]

    # Concatenate totale
    X_total = np.concatenate([normal_windows, positive_windows], axis=0)
    y_total = np.concatenate([y_normal, y_positive], axis=0)

    # --- Split test finale ---
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_total, y_total, test_size=test_split, stratify=y_total, random_state=42
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Crea fold su train+val ---
    folds = []
    if k_fold is not None:
        skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
        for train_idx, val_idx in skf.split(X_trainval, y_trainval):
            X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
            y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

            train_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32)
            )
            val_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32)
            )
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            folds.append((train_loader, val_loader))

    return folds, test_loader
