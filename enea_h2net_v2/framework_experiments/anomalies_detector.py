import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from build_supervised_dataset import *


class LSTMDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


    
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

    def _train_val_loop(self, train_loader, val_loader, num_epochs, optimizer, criterion, device):
        def compute_metrics(all_preds, all_labels):
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()
            pred_labels = (all_preds >= 0.5).astype(int)
            acc = accuracy_score(all_labels, pred_labels)
            prec = precision_score(all_labels, pred_labels, zero_division=0)
            rec = recall_score(all_labels, pred_labels, zero_division=0)
            f1 = f1_score(all_labels, pred_labels, zero_division=0)
            auc = roc_auc_score(all_labels, all_preds)
            return acc, prec, rec, f1, auc

        all_preds_val_fold = []
        all_labels_val_fold = []

        for epoch in range(num_epochs):
            # --- Training ---
            self.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
                optimizer.zero_grad()
                output = self(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
            train_loss /= len(train_loader.dataset)

            # --- Validation ---
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
                    output = self(batch_X)
                    val_loss += criterion(output, batch_y).item() * batch_X.size(0)
                    all_preds_val_fold.append(output.cpu())
                    all_labels_val_fold.append(batch_y.cpu())
            val_loss /= len(val_loader.dataset)

            acc, prec, rec, f1, auc = compute_metrics(all_preds_val_fold, all_labels_val_fold)
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, Acc: {acc:.4f}, Prec: {prec:.4f}, "
                  f"Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        # return metrics del fold
        return compute_metrics(all_preds_val_fold, all_labels_val_fold)

    def train_and_test_model(self, folds=None, test_loader=None, num_epochs=100, lr=1e-4):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        if folds is not None:
            all_fold_metrics = []
            for fold_idx, (train_loader_fold, val_loader_fold) in enumerate(folds, 1):
                print(f"\n=== Fold {fold_idx}/{len(folds)} ===")
                metrics = self._train_val_loop(train_loader_fold, val_loader_fold, num_epochs, optimizer, criterion, device)
                all_fold_metrics.append(metrics)

            # --- Media e std metriche ---
            all_fold_metrics = np.array(all_fold_metrics)
            mean_metrics = all_fold_metrics.mean(axis=0)
            std_metrics = all_fold_metrics.std(axis=0)
            print(f"\n=== Cross-Validation Mean Metrics ===")
            print(f"Acc: {mean_metrics[0]:.4f} ± {std_metrics[0]:.4f}, "
                  f"Prec: {mean_metrics[1]:.4f} ± {std_metrics[1]:.4f}, "
                  f"Rec: {mean_metrics[2]:.4f} ± {std_metrics[2]:.4f}, "
                  f"F1: {mean_metrics[3]:.4f} ± {std_metrics[3]:.4f}, "
                  f"AUC: {mean_metrics[4]:.4f} ± {std_metrics[4]:.4f}")

        if test_loader is not None:
            # --- Test finale ---
            self.eval()
            test_loss = 0
            all_preds_test = []
            all_labels_test = []
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
                    output = self(batch_X)
                    test_loss += criterion(output, batch_y).item() * batch_X.size(0)
                    all_preds_test.append(output.cpu())
                    all_labels_test.append(batch_y.cpu())
            test_loss /= len(test_loader.dataset)
            all_preds_test = torch.cat(all_preds_test).numpy()
            all_labels_test = torch.cat(all_labels_test).numpy()
            pred_labels = (all_preds_test >= 0.5).astype(int)
            acc = accuracy_score(all_labels_test, pred_labels)
            prec = precision_score(all_labels_test, pred_labels, zero_division=0)
            rec = recall_score(all_labels_test, pred_labels, zero_division=0)
            f1 = f1_score(all_labels_test, pred_labels, zero_division=0)
            auc = roc_auc_score(all_labels_test, all_preds_test)
            print(f"\nFinal Test - Loss: {test_loss:.6f}, Acc: {acc:.4f}, Prec: {prec:.4f}, "
                  f"Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")


class ConvLSTM_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, lstm_dropout=0.2, cnn_channels=32, kernel_size=3, cnn_dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=cnn_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Dropout(cnn_dropout)
        )
        self.lstm = nn.LSTM(cnn_channels, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(lstm_dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

    def _train_val_loop(self, train_loader, val_loader, num_epochs, optimizer, criterion, device):
        # stesso loop della LSTM
        all_preds_val_fold, all_labels_val_fold = [], []

        def compute_metrics(all_preds, all_labels):
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()
            pred_labels = (all_preds >= 0.5).astype(int)
            acc = accuracy_score(all_labels, pred_labels)
            prec = precision_score(all_labels, pred_labels, zero_division=0)
            rec = recall_score(all_labels, pred_labels, zero_division=0)
            f1 = f1_score(all_labels, pred_labels, zero_division=0)
            auc = roc_auc_score(all_labels, all_preds)
            return acc, prec, rec, f1, auc

        for epoch in range(num_epochs):
            # --- Training ---
            self.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
                optimizer.zero_grad()
                output = self(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
            train_loss /= len(train_loader.dataset)

            # --- Validation ---
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
                    output = self(batch_X)
                    val_loss += criterion(output, batch_y).item() * batch_X.size(0)
                    all_preds_val_fold.append(output.cpu())
                    all_labels_val_fold.append(batch_y.cpu())
            val_loss /= len(val_loader.dataset)

            acc, prec, rec, f1, auc = compute_metrics(all_preds_val_fold, all_labels_val_fold)
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, Acc: {acc:.4f}, Prec: {prec:.4f}, "
                  f"Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        return compute_metrics(all_preds_val_fold, all_labels_val_fold)

    def train_and_test_model(self, folds=None, test_loader=None, num_epochs=100, lr=1e-4):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if folds is not None:
            all_fold_metrics = []
            for fold_idx, (train_loader_fold, val_loader_fold) in enumerate(folds, 1):
                print(f"\n=== Fold {fold_idx}/{len(folds)} ===")
                metrics = self._train_val_loop(train_loader_fold, val_loader_fold, num_epochs, optimizer, criterion, device)
                all_fold_metrics.append(metrics)

            all_fold_metrics = np.array(all_fold_metrics)
            mean_metrics = all_fold_metrics.mean(axis=0)
            std_metrics = all_fold_metrics.std(axis=0)
            print(f"\n=== Cross-Validation Mean Metrics ===")
            print(f"Acc: {mean_metrics[0]:.4f} ± {std_metrics[0]:.4f}, "
                  f"Prec: {mean_metrics[1]:.4f} ± {std_metrics[1]:.4f}, "
                  f"Rec: {mean_metrics[2]:.4f} ± {std_metrics[2]:.4f}, "
                  f"F1: {mean_metrics[3]:.4f} ± {std_metrics[3]:.4f}, "
                  f"AUC: {mean_metrics[4]:.4f} ± {std_metrics[4]:.4f}")

        if test_loader is not None:
            self.eval()
            all_preds_test, all_labels_test = [], []
            test_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
                    output = self(batch_X)
                    test_loss += criterion(output, batch_y).item() * batch_X.size(0)
                    all_preds_test.append(output.cpu())
                    all_labels_test.append(batch_y.cpu())
            test_loss /= len(test_loader.dataset)
            all_preds_test = torch.cat(all_preds_test).numpy()
            all_labels_test = torch.cat(all_labels_test).numpy()
            pred_labels = (all_preds_test >= 0.5).astype(int)
            acc = accuracy_score(all_labels_test, pred_labels)
            prec = precision_score(all_labels_test, pred_labels, zero_division=0)
            rec = recall_score(all_labels_test, pred_labels, zero_division=0)
            f1 = f1_score(all_labels_test, pred_labels, zero_division=0)
            auc = roc_auc_score(all_labels_test, all_preds_test)
            print(f"\nFinal Test - Loss: {test_loss:.6f}, Acc: {acc:.4f}, Prec: {prec:.4f}, "
                  f"Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")



class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def _train_val_loop(self, train_loader, val_loader, num_epochs, optimizer, criterion, device):
        # esattamente come nella LSTM
        all_preds_val_fold, all_labels_val_fold = [], []
        def compute_metrics(all_preds, all_labels):
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()
            pred_labels = (all_preds >= 0.5).astype(int)
            acc = accuracy_score(all_labels, pred_labels)
            prec = precision_score(all_labels, pred_labels, zero_division=0)
            rec = recall_score(all_labels, pred_labels, zero_division=0)
            f1 = f1_score(all_labels, pred_labels, zero_division=0)
            auc = roc_auc_score(all_labels, all_preds)
            return acc, prec, rec, f1, auc

        for epoch in range(num_epochs):
            self.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
                optimizer.zero_grad()
                output = self(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
            train_loss /= len(train_loader.dataset)

            self.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
                    output = self(batch_X)
                    val_loss += criterion(output, batch_y).item() * batch_X.size(0)
                    all_preds_val_fold.append(output.cpu())
                    all_labels_val_fold.append(batch_y.cpu())
            val_loss /= len(val_loader.dataset)
            acc, prec, rec, f1, auc = compute_metrics(all_preds_val_fold, all_labels_val_fold)
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, Acc: {acc:.4f}, Prec: {prec:.4f}, "
                  f"Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        return compute_metrics(all_preds_val_fold, all_labels_val_fold)

    def train_and_test_model(self, folds=None, test_loader=None, num_epochs=100, lr=1e-4):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if folds is not None:
            all_fold_metrics = []
            for fold_idx, (train_loader_fold, val_loader_fold) in enumerate(folds, 1):
                print(f"\n=== Fold {fold_idx}/{len(folds)} ===")
                metrics = self._train_val_loop(train_loader_fold, val_loader_fold, num_epochs, optimizer, criterion, device)
                all_fold_metrics.append(metrics)

            # Media e std metriche fold
            all_fold_metrics = np.array(all_fold_metrics)
            mean_metrics = all_fold_metrics.mean(axis=0)
            std_metrics = all_fold_metrics.std(axis=0)
            print(f"\n=== Cross-Validation Mean Metrics ===")
            print(f"Acc: {mean_metrics[0]:.4f} ± {std_metrics[0]:.4f}, "
                  f"Prec: {mean_metrics[1]:.4f} ± {std_metrics[1]:.4f}, "
                  f"Rec: {mean_metrics[2]:.4f} ± {std_metrics[2]:.4f}, "
                  f"F1: {mean_metrics[3]:.4f} ± {std_metrics[3]:.4f}, "
                  f"AUC: {mean_metrics[4]:.4f} ± {std_metrics[4]:.4f}")

        # Test finale
        if test_loader is not None:
            self.eval()
            all_preds_test, all_labels_test = [], []
            test_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
                    output = self(batch_X)
                    test_loss += criterion(output, batch_y).item() * batch_X.size(0)
                    all_preds_test.append(output.cpu())
                    all_labels_test.append(batch_y.cpu())
            test_loss /= len(test_loader.dataset)
            all_preds_test = torch.cat(all_preds_test).numpy()
            all_labels_test = torch.cat(all_labels_test).numpy()
            pred_labels = (all_preds_test >= 0.5).astype(int)
            acc = accuracy_score(all_labels_test, pred_labels)
            prec = precision_score(all_labels_test, pred_labels, zero_division=0)
            rec = recall_score(all_labels_test, pred_labels, zero_division=0)
            f1 = f1_score(all_labels_test, pred_labels, zero_division=0)
            auc = roc_auc_score(all_labels_test, all_preds_test)
            print(f"\nFinal Test - Loss: {test_loss:.6f}, Acc: {acc:.4f}, Prec: {prec:.4f}, "
                  f"Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

     
