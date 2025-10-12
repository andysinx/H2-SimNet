import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import os

# -----------------------
# Utility: Dataset -> DataLoader
# -----------------------
def make_loader(X, y=None, batch_size=64, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    if y is not None:
        y_t = torch.tensor(y, dtype=torch.float32)
        ds = torch.utils.data.TensorDataset(X_t, y_t)
    else:
        ds = torch.utils.data.TensorDataset(X_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# -----------------------
# LSTM Autoencoder
# -----------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size=None, dropout_rate=0.0):
        super().__init__()
        if latent_size is None:
            latent_size = hidden_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        # Encoder
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        # Decoder
        self.dec_fc = nn.Linear(hidden_size, latent_size)
        self.out_fc = nn.Linear(latent_size, input_size)  # ricostruisce le feature originali

    def forward(self, x):
        # x: (B, T, F)
        enc_out, _ = self.encoder_lstm(x)            # (B, T, H)
        z = enc_out[:, -1, :]                        # (B, H)
        z = self.dropout(z)
        dec_in = torch.relu(self.dec_fc(z))          # (B, latent)
        out = self.out_fc(dec_in)                    # (B, F)
        T = x.size(1)
        recon = out.unsqueeze(1).repeat(1, T, 1)    # (B, T, F)
        return recon


# -----------------------
# AE Autoencoder (flatten -> bottleneck -> reconstruct)
# -----------------------
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



# -----------------------
# CNN+LSTM Autoencoder
# -----------------------

class ConvLSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size=None, dropout_rate=0.0, kernel_size=3):
        super().__init__()
        if latent_size is None:
            latent_size = hidden_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        
        # Conv layer: estrazione caratteristiche temporali
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Decoder
        self.dec_fc = nn.Linear(hidden_size, latent_size)
        self.out_fc = nn.Linear(latent_size, input_size)

    def forward(self, x):
        # x: (B, T, F)
        x_conv = x.permute(0, 2, 1)          # (B, F, T) per Conv1d
        x_conv = self.relu(self.conv1(x_conv))
        x_conv = x_conv.permute(0, 2, 1)     # (B, T, F) per LSTM
        
        enc_out, _ = self.encoder_lstm(x_conv)  # (B, T, H)
        z = enc_out[:, -1, :]                   # (B, H)
        z = self.dropout(z)
        dec_in = torch.relu(self.dec_fc(z))     # (B, latent)
        out = self.out_fc(dec_in)               # (B, F)
        
        T = x.size(1)
        recon = out.unsqueeze(1).repeat(1, T, 1)  # (B, T, F)
        return recon
# -----------------------
# Training / validation / test loop (unsupervised)
# -----------------------
def compute_reconstruction_scores(model, loader, device, reduction='mean'):
    model.eval()
    losses = []
    all_labels = []
    criterion = nn.MSELoss(reduction='none')  # to get per-element errors
    with torch.no_grad():
        for batch in loader:
            X = batch[0].to(device)
            recon = model(X)
            err = criterion(recon, X)  # (B, T, F)
            # reduce per sample
            if reduction == 'mean':
                sample_err = err.reshape(err.size(0), -1).mean(dim=1)
            elif reduction == 'sum':
                sample_err = err.reshape(err.size(0), -1).sum(dim=1)
            else:
                sample_err = err.reshape(err.size(0), -1).mean(dim=1)
            losses.append(sample_err.cpu().numpy())
            # labels if present
            if len(batch) > 1:
                all_labels.append(batch[1].cpu().numpy())
    scores = np.concatenate(losses, axis=0)
    labels = np.concatenate(all_labels, axis=0) if all_labels else None
    return labels, scores


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
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
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
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}")

    # restore best weights if available
    if val_loader is not None and 'best_state' in locals():
        for k, v in best_state.items():
            best_state[k] = v.to(device)
        model.load_state_dict(best_state)

    return model


'''def evaluate_unsupervised_cost(model, test_loader, device=None, reduction='mean', alpha=0.8):
    """
    Valuta l'autoencoder su dati di test e calcola threshold minimizzando 
    costo operativo FP + alpha * FN.
    
    alpha < 1: FN pesa meno; alpha > 1: FN pesa di più.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    labels, scores = compute_reconstruction_scores(model, test_loader, device, reduction=reduction)
    
    if labels is None:
        # solo score disponibili
        return scores
    
    # calcolo threshold ottimale
    fpr, tpr, thr = roc_curve(labels, scores)
    costs = []
    for t in thr:
        preds = (scores >= t).astype(int)
        FP = np.sum((preds==1) & (labels==0))
        FN = np.sum((preds==0) & (labels==1))
        costs.append(FP + alpha*FN)
    best_idx = np.argmin(costs)
    best_thr = thr[best_idx]
    
    # predizioni finali
    pred_labels = (scores >= best_thr).astype(int)
    
    # metriche classiche (per info)
    acc = balanced_accuracy_score(labels, pred_labels)
    prec = precision_score(labels, pred_labels, zero_division=0)
    rec = recall_score(labels, pred_labels, zero_division=0)
    f1 = f1_score(labels, pred_labels, zero_division=0)
    
    print(f"Semi Supervised Eval (FP + {alpha}*FN) -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, Threshold: {best_thr:.6f}")
    
    return labels, scores, {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'threshold': best_thr}'''

'''def evaluate_unsupervised_cost(model, test_loader, file_name, device=None,
                               reduction='mean',
                               smoothing_window=5,
                               percentiles=[10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99],
                               output_txt="analysis_res.txt"):
    """
    Valuta l'autoencoder su dati di test usando diversi percentili come threshold.
    Salva i risultati in un file txt.

    - smoothing_window: media mobile sui reconstruction scores (intero)
    - percentiles: lista di percentili da valutare
    - file_name: nome del file di origine dei dati (usato per stampa/salvataggio)
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    labels, scores = compute_reconstruction_scores(model, test_loader, device, reduction=reduction)

    if labels is None:
        return scores

    # smoothing opzionale
    if smoothing_window and smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        scores = np.convolve(scores, kernel, mode='same')

    results = []

    for pct in percentiles:
        threshold = np.percentile(scores, pct)
        pred_labels = (scores >= threshold).astype(int)

        metrics = {
            'file': file_name,
            'model': "MPL" if isinstance(model, Autoencoder) else ("LSTM" if isinstance(model, LSTMAutoencoder) else "Unknown"),
            'percentile': pct,
            'threshold': float(threshold),
            'FP': int(np.sum((pred_labels == 1) & (labels == 0))),
            'FN': int(np.sum((pred_labels == 0) & (labels == 1))),
            'bal_acc': float(balanced_accuracy_score(labels, pred_labels)),
            'prec': float(precision_score(labels, pred_labels, zero_division=0)),
            'rec': float(recall_score(labels, pred_labels, zero_division=0)),
            'f1': float(f1_score(labels, pred_labels, zero_division=0)),
            'auc': float(roc_auc_score(labels, scores))
        }
        mod = "MPL" if isinstance(model, Autoencoder) else "LSTM"
        results.append(metrics)
        print(f"{file_name}| {mod} | Percentile {pct}: Threshold: {metrics['threshold']:.4f}, "
              f"FP: {metrics['FP']}, FN: {metrics['FN']}, Bal_Acc: {metrics['bal_acc']:.4f}, "
              f"Prec: {metrics['prec']:.4f}, Rec: {metrics['rec']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")

    # --- salva su file txt ---
    header = "file,percentile,threshold,FP,FN,Bal_Acc,Prec,Rec,F1,AUC\n"
    file_exists = os.path.exists(output_txt)
    with open(output_txt, "a") as f:
        if not file_exists:
            f.write(header)
        for m in results:
            line = f"{m['file']},{m['percentile']},{m['threshold']:.4f},{m['FP']},{m['FN']},{m['bal_acc']:.4f},{m['prec']:.4f},{m['rec']:.4f},{m['f1']:.4f},{m['auc']:.4f}\n"
            f.write(line)

    return results'''



def evaluate_unsupervised_cost(model, test_loader, file_name, device=None,
                               reduction='mean',
                               smoothing_window=5,
                               percentiles=[10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99],
                               output_txt="analysis_res.txt"):
    import seaborn as sns
    """
    Valuta l'autoencoder su dati di test usando diversi percentili come threshold.
    Salva i risultati in un file txt e la curva ROC come immagine .png.
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    labels, scores = compute_reconstruction_scores(model, test_loader, device, reduction=reduction)

    model_name = "AE" if isinstance(model, Autoencoder) else ("LSTM" if isinstance(model, LSTMAutoencoder) else "CNN_LSTM")
    # --- salva i reconstruction scores per riutilizzo ---
    os.makedirs("./result", exist_ok=True)
    base_name = os.path.basename(file_name).replace(".parquet","")

    np.save(f"./result/scores_{model_name}_{base_name}.npy", scores)
    if labels is not None:
        np.save(f"./result/labels_{model_name}_{base_name}.npy", labels)

    # Percentile scelto come “best threshold” (puoi usare lo stesso che usi nel FP/FN plot)
    '''best_percentile = 20
    best_threshold = np.percentile(scores, best_percentile)

    # === Plot con seaborn ===
    sns.set(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=(7,5))

    # Istogramma + KDE
    sns.histplot(scores, bins=30, kde=True, color="steelblue", stat="density", alpha=0.8)

    # Linea verticale del threshold “best”
    plt.axvline(best_threshold, color="red", linestyle="--", linewidth=2)

    # Etichetta best
    plt.text(best_threshold + 0.001, plt.ylim()[1]*0.9, f"best ({best_percentile}° pct)", 
            color="red", fontsize=10, weight="bold")'''
    
    
    
    '''distriberr_path = f"./result/distributerr_{model_name}_{os.path.basename(file_name).replace('.parquet','')}.pdf"
    # Titolo e label
    plt.title(f"Distribution of Anomaly Scores — {model_name}", fontsize=13, weight="bold")
    plt.xlabel("Reconstruction error")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(distriberr_path, dpi=300)'''

    if labels is None:
        return scores

    # smoothing opzionale
    if smoothing_window and smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        scores = np.convolve(scores, kernel, mode='same')

    results = []

    # Calcolo ROC e AUC (unici per ogni modello/file)
    fpr, tpr, _ = roc_curve(labels, scores)
    auc_value = roc_auc_score(labels, scores)
    print(auc_value)

    # --- salva la curva ROC ---
    roc_path = f"./result/roc_{model_name}_{os.path.basename(file_name).replace('.parquet','')}.pdf"
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_value:.4f})')
    plt.plot([0,1], [0,1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC - {model_name} - {file_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # --- calcolo metriche per vari percentili ---
    for pct in percentiles:
        threshold = np.percentile(scores, pct)
        pred_labels = (scores >= threshold).astype(int)
        metrics = {
            'file': file_name,
            'model': model_name,
            'percentile': pct,
            'threshold': float(threshold),
            'FP': int(np.sum((pred_labels == 1) & (labels == 0))),
            'FN': int(np.sum((pred_labels == 0) & (labels == 1))),
            'bal_acc': balanced_accuracy_score(labels, pred_labels),
            'prec': precision_score(labels, pred_labels, zero_division=0),
            'rec': recall_score(labels, pred_labels, zero_division=0),
            'f1': f1_score(labels, pred_labels, zero_division=0),
            'auc': auc_value
        }

        results.append(metrics)
        print(f"{file_name} | {model_name} | Percentile {pct}: Threshold {metrics['threshold']:.4f}, "
              f"FP {metrics['FP']}, FN {metrics['FN']}, Bal_Acc {metrics['bal_acc']:.4f}, "
              f"Prec {metrics['prec']:.4f}, Rec {metrics['rec']:.4f}, F1 {metrics['f1']:.4f}, AUC {metrics['auc']:.4f}")

    # --- salva su file txt ---
    header = "file,model,percentile,threshold,FP,FN,Bal_Acc,Prec,Rec,F1,AUC\n"
    file_exists = os.path.exists(output_txt)
    with open(output_txt, "a") as f:
        if not file_exists:
            f.write(header)
        for m in results:
            f.write(f"{m['file']},{m['model']},{m['percentile']},{m['threshold']:.4f},"
                    f"{m['FP']},{m['FN']},{m['bal_acc']:.4f},{m['prec']:.4f},"
                    f"{m['rec']:.4f},{m['f1']:.4f},{m['auc']:.4f}\n")

    print(f"ROC curve saved to {roc_path}")
    return results
