from build_unsupervised_dataset import *
from anomalies_detector import *

if __name__ == "__main__":
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

    output_txt = "analysis_res.txt"
    if os.path.exists(output_txt):
        os.remove(output_txt)

    # --- Accumulare risultati per scenario ---
    results_per_file = []

    for file_path in files_ordered:
        print(f"\n=== Experiment on {file_path} ===")
        anomalous_file = "../anomalous-scenarios/parquet/" + file_path
        first_part = file_path.split("/")[0]

        folds, test_loader = create_unsupervised_dataset(
            anomalous_file=anomalous_file,
            position=first_part,
            window_size=100,
            derivatives=False,
            k_fold=5,
            batch_size=256,
            test_split=0.1,
            step=50
        )   

        sample_X, _ = next(iter(folds[0][0]))
        seq_len = sample_X.shape[1]
        input_size = sample_X.shape[2]

        scenario_results = {"file": file_path}

        # --- MLP Autoencoder ---
        print("Autoencoder:")   
        model_ae = Autoencoder(seq_len=seq_len, input_size=input_size, latent_size=32)
        for train_loader, val_loader in folds:
            model_ae = train_autoencoder(model_ae, train_loader, val_loader, num_epochs=100, lr=1e-3, early_stopping=None)
        results = evaluate_unsupervised_cost(model_ae, test_loader,file_name=file_path)

        # --- LSTM Autoencoder ---
        print("LSTM Autoencoder:")
        model_lstm = LSTMAutoencoder(input_size=input_size, hidden_size=64, latent_size=32)
        for train_loader, val_loader in folds:
            model_lstm = train_autoencoder(model_lstm, train_loader, val_loader, num_epochs=100, lr=1e-3, early_stopping=None)
        results = evaluate_unsupervised_cost(model_lstm, test_loader, file_name=file_path)

        # --- CNN + LSTM Autoencoder ---
        print("CNN + LSTM Autoencoder:")
        model_cnn_lstm = ConvLSTMAutoencoder(input_size=input_size, hidden_size=64, latent_size=32, kernel_size=3)
        for train_loader, val_loader in folds:
            model_cnn_lstm = train_autoencoder(model_cnn_lstm, train_loader, val_loader, num_epochs=100, lr=1e-3, early_stopping=None)
        results = evaluate_unsupervised_cost(model_cnn_lstm, test_loader, file_name=file_path)