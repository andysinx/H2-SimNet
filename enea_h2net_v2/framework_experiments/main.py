from build_supervised_dataset import *
from anomalies_detector import *


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
            batch_size=256             # numero di fold desiderato
        )

        # Determiniamo dimensioni input
        sample_X, _ = next(iter(folds[0][0]))  # primo batch del primo fold
        seq_len = sample_X.shape[1]
        input_size = sample_X.shape[2]
        # --- 1️⃣ MLP ---
        print('MLP RESULT:')
        model_1 = MLPClassifier(input_size=input_size * seq_len, hidden_size=64, dropout_rate=0.2)
        model_1.train_and_test_model(folds=folds, num_epochs=70, lr=1e-4, test_loader=test_loader)

        # --- 2️⃣ LSTM ---
        print('LSTM RESULT:')
        model = LSTMClassifier(input_size=input_size, hidden_size=64, dropout_rate=0.2)
        model.train_and_test_model(folds=folds, num_epochs=70, lr=1e-4, test_loader=test_loader)

        # --- 3️⃣ CNN + LSTM ---
        print('CNN + LSTM RESULT:')
        model_3 = ConvLSTM_Classifier(input_size=input_size, hidden_size=64, lstm_dropout=0.2, 
                                     cnn_channels=32, kernel_size=3, cnn_dropout=0.2)
        model_3.train_and_test_model(folds=folds, num_epochs=70, lr=1e-4, test_loader=test_loader)