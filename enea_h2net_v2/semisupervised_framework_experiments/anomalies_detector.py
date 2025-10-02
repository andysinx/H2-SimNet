import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

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
    def __init__(self, input_size, hidden_size, latent_size=None, dropout_rate=0.2):
        super().__init__()
        if latent_size is None:
            latent_size = hidden_size
        # Encoder
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.enc_fc = nn.Linear(hidden_size, latent_size)
        # Decoder
        self.dec_fc = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(hidden_size, input_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (B, T, F)
        enc_out, _ = self.encoder_lstm(x)         # (B, T, H)
        # take last time-step representation
        z = enc_out[:, -1, :]                     # (B, H)
        z = self.enc_fc(z)                        # (B, latent)
        z = torch.relu(z)
        z = self.dropout(z)
        # expand z to sequence length for decoder
        dec_in = self.dec_fc(z)                   # (B, H)
        # repeat across time steps
        T = x.size(1)
        dec_in_seq = dec_in.unsqueeze(1).repeat(1, T, 1)  # (B, T, H)
        dec_out, _ = self.decoder_lstm(dec_in_seq)       # (B, T, input_size)
        recon = dec_out
        return recon


# -----------------------
# ConvLSTM Autoencoder (1D conv + LSTM decoder)
# -----------------------
class ConvLSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size=64, cnn_channels=32, kernel_size=3, latent_size=None, dropout=0.2):
        super().__init__()
        if latent_size is None:
            latent_size = hidden_size
        # encoder conv -> lstm -> latent
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=cnn_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.encoder_lstm = nn.LSTM(cnn_channels, hidden_size, batch_first=True)
        self.enc_fc = nn.Linear(hidden_size, latent_size)
        # decoder: latent -> fc -> lstm -> conv-transpose like output
        self.dec_fc = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(hidden_size, cnn_channels, batch_first=True)
        # map cnn_channels back to original feature dim via 1x1 conv
        self.out_conv = nn.Conv1d(in_channels=cnn_channels, out_channels=input_size, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, F)
        B, T, F = x.shape
        x_c = x.permute(0, 2, 1)            # (B, F, T)
        c_out = self.conv(x_c)              # (B, C, T)
        c_out = c_out.permute(0, 2, 1)      # (B, T, C)
        enc_out, _ = self.encoder_lstm(c_out)
        z = enc_out[:, -1, :]               # (B, H)
        z = self.enc_fc(z)                  # (B, latent)
        z = torch.relu(z)
        z = self.dropout(z)
        dec_in = self.dec_fc(z)             # (B, H)
        dec_seq = dec_in.unsqueeze(1).repeat(1, T, 1)  # (B, T, H)
        dec_out, _ = self.decoder_lstm(dec_seq)       # (B, T, C)
        dec_out = dec_out.permute(0, 2, 1)             # (B, C, T)
        recon = self.out_conv(dec_out)                 # (B, F, T)
        recon = recon.permute(0, 2, 1)                 # (B, T, F)
        return recon


# -----------------------
# MLP Autoencoder (flatten -> bottleneck -> reconstruct)
# -----------------------
class MLPAutoencoder(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size=64, latent_size=32, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        in_dim = seq_len * input_size
        self.enc_fc1 = nn.Linear(in_dim, hidden_size)
        self.enc_fc2 = nn.Linear(hidden_size, latent_size)
        self.dec_fc1 = nn.Linear(latent_size, hidden_size)
        self.dec_fc2 = nn.Linear(hidden_size, in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, F)
        B = x.size(0)
        flat = x.view(B, -1)
        h = torch.relu(self.enc_fc1(flat))
        h = self.dropout(h)
        z = torch.relu(self.enc_fc2(h))
        h2 = torch.relu(self.dec_fc1(z))
        h2 = self.dropout(h2)
        out_flat = self.dec_fc2(h2)
        recon = out_flat.view(B, self.seq_len, self.input_size)
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


def evaluate_unsupervised_cost(model, test_loader, device=None, reduction='mean', alpha=0.6):
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
    
    return labels, scores, {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'threshold': best_thr}

'''def evaluate_unsupervised_cost(model, test_loader, device=None, reduction='mean',
                                           alpha_values=None):
    """
    Valuta l'autoencoder su dati di test e calcola threshold minimizzando 
    costo operativo FP + alpha * FN per diversi valori di alpha.
    
    Restituisce le metriche e i threshold ottimali per ciascun alpha.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    labels, scores = compute_reconstruction_scores(model, test_loader, device, reduction=reduction)
    
    if labels is None:
        return scores
    
    if alpha_values is None:
        alpha_values = np.arange(0.5, 1.01, 0.1)
    
    results = {}
    
    fpr, tpr, thr = roc_curve(labels, scores)
    
    for alpha in alpha_values:
        costs = []
        for t in thr:
            preds = (scores >= t).astype(int)
            FP = np.sum((preds==1) & (labels==0))
            FN = np.sum((preds==0) & (labels==1))
            costs.append(FP + alpha*FN)
        
        best_idx = np.argmin(costs)
        best_thr = thr[best_idx]
        pred_labels = (scores >= best_thr).astype(int)
        
        results[alpha] = {
            'threshold': best_thr,
            'FP': np.sum((pred_labels==1) & (labels==0)),
            'FN': np.sum((pred_labels==0) & (labels==1)),
            'acc': balanced_accuracy_score(labels, pred_labels),
            'prec': precision_score(labels, pred_labels, zero_division=0),
            'rec': recall_score(labels, pred_labels, zero_division=0),
            'f1': f1_score(labels, pred_labels, zero_division=0)
        }
        
        print(f"Alpha={alpha:.1f} -> Threshold: {best_thr:.4f}, FP: {results[alpha]['FP']}, "
              f"FN: {results[alpha]['FN']}, Acc: {results[alpha]['acc']:.4f}, "
              f"Prec: {results[alpha]['prec']:.4f}, Rec: {results[alpha]['rec']:.4f}, "
              f"F1: {results[alpha]['f1']:.4f}")
    
    return labels, scores, results'''
