from build_unsupervised_dataset import *
from anomalies_detector import *

if __name__ == "__main__":
    files_ordered = [
        "assestment_transient/leakG4.parquet",
        "assestment_transient/leakG5G6_overlapped.parquet",
        "assestment_transient/leakG5G6_overlapped_noisy.parquet",
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

    # --- Accumulare risultati per scenario ---
    results_per_file = []

    for file_path in files_ordered:
        print(f"\n=== Experiment on {file_path} ===")
        anomalous_file = "../anomalous-scenarios/parquet/" + file_path

        folds, test_loader = create_unsupervised_dataset(
            anomalous_file=anomalous_file,
            window_size=10,
            step=1,
            derivatives=False,
            k_fold=5,
            batch_size=256,
            max_anom_ratio=0.05
        )

        sample_X, _ = next(iter(folds[0][0]))
        seq_len = sample_X.shape[1]
        input_size = sample_X.shape[2]

        scenario_results = {"file": file_path}

        # --- MLP Autoencoder ---
        print("MPL Autoencoder:")
        model_mlp = MLPAutoencoder(seq_len=seq_len, input_size=input_size, hidden_size=64, latent_size=32)
        for train_loader, val_loader in folds:
            model_mlp = train_autoencoder(model_mlp, train_loader, val_loader, num_epochs=50, lr=1e-3, early_stopping=None)
        labels, scores, metrics = evaluate_unsupervised_cost(model_mlp, test_loader)
        scenario_results["MLP"] = metrics

        # --- LSTM Autoencoder ---
        print("LSTM Autoencoder:")
        model_lstm = LSTMAutoencoder(input_size=input_size, hidden_size=64, latent_size=32)
        for train_loader, val_loader in folds:
            model_lstm = train_autoencoder(model_lstm, train_loader, val_loader, num_epochs=50, lr=1e-3, early_stopping=None)
        labels, scores, metrics = evaluate_unsupervised_cost(model_lstm, test_loader)
        scenario_results["LSTM"] = metrics

        results_per_file.append(scenario_results)

    # --- Stampare risultati finali ---
    for res in results_per_file:
        print(f"\nScenario: {res['file']}")
        for model_name in ["MLP", "LSTM", "CNN+LSTM"]:
            m = res[model_name]
            print(f"{model_name} -> AUC: {m['auc']:.4f}, Acc: {m['acc']:.4f}, Prec: {m['prec']:.4f}, Rec: {m['rec']:.4f}, F1: {m['f1']:.4f}, Thr: {m['threshold']:.4f}")
