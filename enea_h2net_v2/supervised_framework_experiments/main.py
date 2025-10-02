from build_supervised_dataset import *
from anomalies_detector import *
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# === Example usage ===
from build_supervised_dataset import *
from anomalies_detector import *
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# === Example usage ===
if __name__ == "__main__":
    files_ordered = [
        # assestment_transient
        "assestment_transient/leakG4.parquet",
        "assestment_transient/leakG5G6_overlapped.parquet",
        "assestment_transient/leakG5G6_overlapped_noisy.parquet",
        # constant_segment
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
        # start_transient
        "start_transient/leakG3.parquet",
        "start_transient/leakG4brokenC1.parquet"
    ]

    # --- Liste per accumulare tutte le predizioni e label ---
    all_labels_mlp, all_preds_mlp = [], []
    all_labels_lstm, all_preds_lstm = [], []
    all_labels_cnnlstm, all_preds_cnnlstm = [], []

    for file_path in files_ordered:
        print(f"\n=== Experiment on {file_path} ===")
        anomalous_file = "../anomalous-scenarios/parquet/" + file_path

        # Otteniamo X, y e opzionalmente folds già pronti
        folds, test_loader = create_balanced_dataset(
            anomalous_file,
            window_size=10,
            step=1,
            derivatives=False,
            position=file_path.partition("/")[0],
            filt_win_size=5,  # nuova opzione in create_balanced_dataset
            k_fold=5,
            batch_size=256
        )

        # Determiniamo dimensioni input
        sample_X, _ = next(iter(folds[0][0]))  # primo batch del primo fold
        seq_len = sample_X.shape[1]
        input_size = sample_X.shape[2]

        # --- 1️⃣ MLP ---
        print('MLP RESULT:')
        model_1 = MLPClassifier(input_size=input_size * seq_len, hidden_size=64, dropout_rate=0.2)
        labels_mlp, preds_mlp = model_1.train_and_test_model(folds=folds, num_epochs=70, lr=1e-4, test_loader=test_loader)
        all_labels_mlp.extend(labels_mlp)
        all_preds_mlp.extend(preds_mlp)

        # --- 2️⃣ LSTM ---
        print('LSTM RESULT:')
        model_lstm = LSTMClassifier(input_size=input_size, hidden_size=64, dropout_rate=0.2)
        labels_lstm, preds_lstm = model_lstm.train_and_test_model(folds=folds, num_epochs=70, lr=1e-4, test_loader=test_loader)
        all_labels_lstm.extend(labels_lstm)
        all_preds_lstm.extend(preds_lstm)

        # --- 3️⃣ CNN + LSTM ---
        print('CNN + LSTM RESULT:')
        model_3 = ConvLSTM_Classifier(input_size=input_size, hidden_size=64, lstm_dropout=0.2, 
                                      cnn_channels=32, kernel_size=3, cnn_dropout=0.2)
        labels_cnnlstm, preds_cnnlstm = model_3.train_and_test_model(folds=folds, num_epochs=70, lr=1e-4, test_loader=test_loader)
        all_labels_cnnlstm.extend(labels_cnnlstm)
        all_preds_cnnlstm.extend(preds_cnnlstm)

    # --- ROC Aggregata per tutti i file ---
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blu, arancio, verde
    plt.figure(figsize=(8,8))

    # MLP
    fpr1, tpr1, _ = roc_curve(all_labels_mlp, all_preds_mlp)
    roc_auc1 = auc(fpr1, tpr1)
    plt.step(fpr1, tpr1, where='post', color=colors[0], lw=3, alpha=0.9, label=f'MLP (AUC={roc_auc1:.2f})', marker='o', markevery=0.1)

    # LSTM
    fpr2, tpr2, _ = roc_curve(all_labels_lstm, all_preds_lstm)
    roc_auc2 = auc(fpr2, tpr2)
    plt.step(fpr2, tpr2, where='post', color=colors[1], lw=3, alpha=0.9, label=f'LSTM (AUC={roc_auc2:.2f})', marker='s', markevery=0.1)

    # CNN+LSTM
    fpr3, tpr3, _ = roc_curve(all_labels_cnnlstm, all_preds_cnnlstm)
    roc_auc3 = auc(fpr3, tpr3)
    plt.step(fpr3, tpr3, where='post', color=colors[2], lw=3, alpha=0.9, label=f'CNN+LSTM (AUC={roc_auc3:.2f})', marker='^', markevery=0.1)

    # Linea diagonale di riferimento
    plt.plot([0,1],[0,1], color='gray', lw=1.5, linestyle='--', alpha=0.7)

    # Stile figure
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve - All Files', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Salvataggio alta qualità
    plt.savefig('./ROC_Curve_Aggregate.pdf', dpi=300)
    plt.close()