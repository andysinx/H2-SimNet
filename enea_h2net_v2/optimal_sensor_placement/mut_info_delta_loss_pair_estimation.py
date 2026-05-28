import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
import torch
import torch.nn as nn
import os
import torch.optim as optim
import itertools
from sklearn.metrics import precision_score, recall_score, f1_score

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
    data = df[feature_cols].values
    labels = df[label_col].values

    windows = []
    y = []

    for i in range(0, len(data) - window_size + 1, step):
        window = data[i:i+window_size]
        window_labels = labels[i:i+window_size]

        # finestra valida SOLO se abbastanza anomalie dentro
        if window_labels.mean() > 0.2:   # almeno 20% anomalie
            windows.append(window)
            y.append(1)

    if len(windows) == 0:
        return np.empty((0, window_size, len(feature_cols))), np.empty(0)

    return np.array(windows, dtype=np.float32), np.array(y, dtype=int)

def load_normal_data(feature_cols, normal_paths, window_size=50, step=1, derivatives=False, filt_win_size=None):
    windows_list = []

    expected_shape = None

    for path in normal_paths:
        df = pd.read_parquet(path)

        if isinstance(filt_win_size, int) and filt_win_size > 1:
            df = apply_moving_average(df, feature_cols, window=filt_win_size)

        if not set(feature_cols).issubset(df.columns):
            print(f"[SKIP NORMAL] missing cols: {path}")
            continue

        w = create_windows(df, feature_cols, window_size, step)

        if derivatives:
            deriv = np.diff(w, axis=1, prepend=w[:, :1, :])
            w = np.concatenate([w, deriv], axis=2)

        if w.shape[0] == 0:
            continue

        # 🔥 lock shape
        if expected_shape is None:
            expected_shape = w.shape[1:]
        elif w.shape[1:] != expected_shape:
            print(f"[SKIP NORMAL] shape mismatch: {path} {w.shape}")
            continue

        windows_list.append(w)

    if len(windows_list) == 0:
        raise ValueError("No valid normal data found")

    windows = np.concatenate(windows_list, axis=0)
    y = np.zeros(len(windows), dtype=int)

    return windows, y

def load_anomalous_data(anomalous_files, feature_cols, window_size=50, step=1,
                        label_col="label_anomaly", position='assestment_transient',
                        derivatives=False):

    topologies = ["anom_gestup"]
    windows_list = []

    for anomalous_file in anomalous_files:

        for topo in topologies:
            full_path = Path(anomalous_file).parent.parent / topo / position / Path(anomalous_file).name

            if not full_path.exists():
                continue

            df = pd.read_parquet(full_path)

            if not set(feature_cols).issubset(df.columns):
                continue

            w, _ = create_positive_windows(df, feature_cols, window_size, step, label_col)

            if w.shape[0] > 0:
                windows_list.append(w)

    if len(windows_list) == 0:
        return np.empty((0, window_size, len(feature_cols))), np.empty(0)

    windows = np.concatenate(windows_list, axis=0)
    y = np.ones(len(windows), dtype=int)

    return windows, y

def generate_ps_pairs():

    ps_cols = [f"PS{i}" for i in range(1, 13)]

    return list(itertools.combinations(ps_cols, 2))

# --- Main unsupervised dataset builder ---
def create_unsupervised_dataset(anomalous_file, position, feature_cols=None, window_size=100, step=1, derivatives=False,
                                 k_fold=None, batch_size=64, test_split=0.2, val_anom_ratio=0.05):
    
    if feature_cols is None:   
        MFS_cols = [f"MFS{i}" for i in range(1, 5)]
        feature_cols = MFS_cols

    # --- Load normal data ---
    normal_paths = [
        #"./H2-SimNet/normality-scenarios/parquet/normality_gest_down.parquet",
        #"./H2-SimNet/normality-scenarios/parquet/normality_gest_center.parquet",
        "./H2-SimNet/normality-scenarios/parquet/normality_gest_up.parquet"
    ]

    # --- Load normal data ---
    clean_paths = [
        #"./H2-SimNet-clean/normality-scenarios/parquet/normality_gest_down.parquet",
        #"./H2-SimNet-clean/normality-scenarios/parquet/normality_gest_center.parquet",
        "./H2-SimNet-clean/normality-scenarios/parquet/normality_gest_up.parquet"
    ]
    all_normal_paths = normal_paths + clean_paths
    np.random.shuffle(all_normal_paths)
    normal_windows, y_normal = load_normal_data(feature_cols, all_normal_paths, window_size, step, derivatives)

    # --- Split train+validation normal vs test normal ---
    X_trainval_norm, X_test_norm, _, _ = train_test_split(normal_windows, y_normal, test_size=test_split, random_state=42)

    # --- Load all anomalous data ---
    X_anom_all, y_anom_all = load_anomalous_data(
        anomalous_files,
        feature_cols,
        window_size=window_size,   # 👈 AGGIUNGI QUESTO
        step=step,
        position=position,
        derivatives=derivatives
    )
    num_val_anom = max(1, int(len(X_anom_all) * val_anom_ratio))
    perm = np.random.permutation(len(X_anom_all))
    X_val_anom = X_anom_all[perm[:num_val_anom]]
    y_val_anom = y_anom_all[perm[:num_val_anom]]
    X_test_anom = X_anom_all[perm[num_val_anom:]]
    y_test_anom = y_anom_all[perm[num_val_anom:]]

    #print("X_anom_all shape:", X_anom_all.shape)
    #print("unique labels:", np.unique(y_anom_all))
    #print("anom ratio:", y_anom_all.mean())

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
            y_train = np.zeros(len(X_train), dtype=int)

            anom_ratio_train = 0.007
            num_anom_train = int(len(X_train) * anom_ratio_train)

            if len(X_anom_all) > 0 and num_anom_train > 0:
                idx = np.random.choice(len(X_anom_all), num_anom_train, replace=True)
                X_train_anom = X_anom_all[idx]

                # aggiungi dati
                X_train = np.concatenate([X_train, X_train_anom], axis=0)

                
                y_train_anom = np.ones(len(X_train_anom), dtype=int)

                # unisci label
                y_train = np.concatenate([y_train, y_train_anom], axis=0)

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


class Autoencoder(nn.Module):
    def __init__(self, seq_len, input_size, latent_size=32, hidden_sizes=[64, 64], dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        in_dim = seq_len * input_size

        # Encoder lineare a più layer
        layers = []
        prev_dim = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, latent_size))  # layer finale encoder
        self.encoder = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

        # Decoder lineare simmetrico
        layers = []
        prev_dim = latent_size
        for h in reversed(hidden_sizes):
            layers.append(nn.Linear(prev_dim, h))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, in_dim))  # layer finale decoder
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        B = x.size(0)
        flat = x.view(B, -1)

        z = self.encoder(flat)
        z = self.dropout(z)
        out_flat = self.decoder(z)
        recon = out_flat.view(B, self.seq_len, self.input_size)
        return recon
        
def train_autoencoder(model, train_loader, val_loader=None, num_epochs=50, lr=1e-3, device=None, weight_decay=0.0, early_stopping=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    best_val_loss = np.inf
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        n = 0
        for batch in train_loader:
            X = batch[0].to(device)
            optimizer.zero_grad()
            recon = model(X)
            loss = criterion(recon, X)
            loss.backward()
            optimizer.step()
            batch_size = X.size(0)
            train_loss += loss.item() * batch_size
            n += batch_size
        train_loss /= n

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            nval = 0
            with torch.no_grad():
                for batch in val_loader:
                    X = batch[0].to(device)
                    recon = model(X)
                    loss = criterion(recon, X)
                    batch_size = X.size(0)
                    val_loss += loss.item() * batch_size
                    nval += batch_size
            val_loss /= nval
            #print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
            # early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # save best weights
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1
                if early_stopping and epochs_no_improve >= early_stopping:
                    print("Early stopping")
                    break
        else:
            #print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}")
            pass
    # restore best weights if available
    if val_loader is not None and 'best_state' in locals():
        for k, v in best_state.items():
            best_state[k] = v.to(device)
        model.load_state_dict(best_state)

    return model


def evaluate_autoencoder(model, test_loader, threshold=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    reconstruction_errors = []
    y_true = []

    criterion = nn.MSELoss(reduction='none')

    with torch.no_grad():
        for batch in test_loader:
            X, y = batch
            X = X.to(device)

            recon = model(X)

            # reconstruction error per sample
            loss = criterion(recon, X)

            # media su time e feature
            loss = loss.mean(dim=(1, 2))

            reconstruction_errors.extend(loss.cpu().numpy())
            y_true.extend(y.numpy())

    reconstruction_errors = np.array(reconstruction_errors)
    y_true = np.array(y_true)

    # threshold automatico se non fornito
    assert threshold is not None, "Threshold must be computed on validation normal set"

    # predizione anomalie
    y_pred = (reconstruction_errors > threshold).astype(int)

    # metriche
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "reconstruction_errors": reconstruction_errors,
        "y_true": y_true,
        "y_pred": y_pred
    }



def compute_LF(model, data_loader, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    criterion = nn.MSELoss(reduction='none')

    losses = []

    with torch.no_grad():
        for batch in data_loader:
            X, _ = batch
            X = X.to(device)

            recon = model(X)

            loss = criterion(recon, X)
            loss = loss.mean(dim=(1, 2))  # L(F) per sample

            losses.extend(loss.cpu().numpy())

    return np.mean(losses), np.array(losses)


def compute_redundancy_matrix(pairs, feature_base, anomalous_files, position, num_runs=5):

    results = {}

    fixed_F = [f"MFS{i}" for i in range(1, 5)]
    WINDOW = 100
    STEP = 50

    for ps_i, ps_j in pairs:

        feature_cols = fixed_F + [ps_i, ps_j]

        LF_runs = []

        for _ in range(num_runs):

            folds, test_loader = create_unsupervised_dataset(
                anomalous_file=anomalous_files,
                position=position,
                feature_cols=feature_cols,
                window_size=WINDOW,
                k_fold=3,
                batch_size=256,
                test_split=0.1,
                step=STEP
            )

            sample_X, _ = next(iter(folds[0][0]))
            seq_len = sample_X.shape[1]
            input_size = sample_X.shape[2]

            model = Autoencoder(
                seq_len=seq_len,
                input_size=input_size
            )

            for train_loader, val_loader in folds:
                model = train_autoencoder(model, train_loader, val_loader, num_epochs=50)

            LF_mean, _ = compute_LF(model, test_loader)
            LF_runs.append(LF_mean)

        results[(ps_i, ps_j)] = {
            "mean": np.mean(LF_runs),
            "std": np.std(LF_runs),
            "var": np.var(LF_runs)
        }

    return results

if __name__ == "__main__":

    NUM_RUNS = 5 

    files_ordered = [
        "assestment_transient/leakG4.parquet",
        "assestment_transient/leakG5G6_overlapped.parquet",
        "constant_segment/brokenC1G5_weak.parquet",
        "constant_segment/brokenC1_strong.parquet",
        "constant_segment/brokenC1_weak.parquet",
        "constant_segment/leakG3.parquet",
        "constant_segment/leakG3G4_strong.parquet",
        "constant_segment/leakG3G4_weak.parquet",
        "constant_segment/leakG3G5_overlapped.parquet",
        "constant_segment/leakG3G6.parquet",
        "constant_segment/leakG4_strong.parquet",
        "constant_segment/leakG4_weak.parquet",
        "constant_segment/leakG5_strong.parquet",
        "constant_segment/leakG5_weak.parquet",
        "constant_segment/leakG6.parquet",
        "constant_segment/startdelayC1.parquet",
        "start_transient/leakG3.parquet",
        "start_transient/leakG4brokenC1.parquet"
    ]

    # ===============================
    # STEP 1: DEFINIZIONE SENSORI
    # ===============================
    fixed_F = [f"MFS{i}" for i in range(1, 5)]
    ps_pairs = generate_ps_pairs()

    # ===============================
    # STEP 2: COSTRUISCI LISTA FILE GLOBALI
    # ===============================
    anomalous_files = [
        "./H2-SimNet/anomalous-scenarios/parquet/" + f
        for f in files_ordered
    ]

    # ===============================
    # STEP 3: REDUNDANCY MATRIX GLOBALE
    # ===============================
    redundancy_matrix = compute_redundancy_matrix(
        pairs=ps_pairs,
        feature_base=fixed_F,
        anomalous_files=anomalous_files,
        position="assestment_transient",
        num_runs=NUM_RUNS
    )

    # ===============================
    # STEP 4: STAMPA RISULTATI
    # ===============================
    results = []

    for (ps_i, ps_j), stats in redundancy_matrix.items():

        print(f"\n========== L(F + {ps_i},{ps_j}) ==========")
        print(f"Mean: {stats['mean']:.6f}")
        print(f"Std:  {stats['std']:.6f}")
        print(f"Var:  {stats['var']:.6f}")

        results.append({
            "pair": (ps_i, ps_j),
            "L_mean": stats["mean"],
            "L_std": stats["std"],
            "L_var": stats["var"]
        })