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
            model_lstm = train_autoencoder(model_lstm, train_loader, val_loader, num_epochs=100, lr=1e-3, early_stopping=None)
        labels, scores, metrics = evaluate_unsupervised_cost(model_lstm, test_loader)
        scenario_results["LSTM"] = metrics

        # --- CNN+LSTM Autoencoder ---
        
        print("CNN + LSTM Autoencoder:")
        model_cnnlstm = ConvLSTMAutoencoder(input_size=input_size, hidden_size=64, cnn_channels=32, latent_size=32)
        for train_loader, val_loader in folds:
            model_cnnlstm = train_autoencoder(model_cnnlstm, train_loader, val_loader, num_epochs=100, lr=1e-3, early_stopping=None)
        labels, scores, metrics = evaluate_unsupervised_cost(model_cnnlstm, test_loader)
        scenario_results["CNN+LSTM"] = metrics

        results_per_file.append(scenario_results)

    # --- Stampare risultati finali ---
    for res in results_per_file:
        print(f"\nScenario: {res['file']}")
        for model_name in ["MLP", "LSTM", "CNN+LSTM"]:
            m = res[model_name]
            print(f"{model_name} -> AUC: {m['auc']:.4f}, Acc: {m['acc']:.4f}, Prec: {m['prec']:.4f}, Rec: {m['rec']:.4f}, F1: {m['f1']:.4f}, Thr: {m['threshold']:.4f}")


'''
Appunti esperimenti: Allora la pipeline e' semisupervised con 5% di anmomalie nel validation per settare il threshold. Ora pero' mi sono accorto che il punto operativo migliore
migliora con la riduzione delle features (ossia rimuovendo la derivata) le metriche migliorano andandoci a perdere qualche falso negativo in piu per qualche falso positivo in piu (non esagerato)
Praticamente circa 29% di falsi positivi risparmiati a costo di ~11% falsi negativi in più (senza derivata).

Nella nostra valutazione semi-supervisionata, abbiamo introdotto un parametro di ponderazione α nel calcolo del threshold dell’autoencoder, definito come 
costo operativo = FP + α·FN, dove FP sono i falsi positivi e FN i falsi negativi. Questo approccio ci ha permesso di esplorare il trade-off tra la 
riduzione dei falsi positivi e il mantenimento delle altre metriche di performance, considerando che nel nostro scenario i falsi positivi comportano un 
costo operativo più elevato. Variando α nell’intervallo da 0.5 a 1.0, abbiamo osservato che per valori di α pari o superiori a 0.6 il threshold ottimale e le 
metriche di accuratezza bilanciata, precisione, recall e F1 si stabilizzano, indicando un punto di equilibrio tra falsi positivi e falsi negativi. La scelta di α = 0.6 
si è rivelata particolarmente efficace, riducendo il numero di falsi positivi senza compromettere in maniera significativa le altre metriche.

Abbiamo inoltre confrontato i risultati ottenuti includendo o escludendo la derivata prima delle serie temporali dei sensori come feature aggiuntiva. 
Rimuovendo la derivata, il numero di falsi negativi è aumentato di circa 22 unità, ma il numero di falsi positivi è diminuito di circa 80 unità rispetto alla configurazione 
con derivata. In termini percentuali, questo corrisponde a un aumento dei falsi negativi di circa il 12% e a una riduzione dei falsi positivi di circa il 28%. Nel complesso, 
la rimozione della derivata ha portato a un miglioramento delle metriche complessive, con un bilanciamento più favorevole tra precisione, recall e F1, confermando che, 
nel nostro scenario, il beneficio derivante dalla riduzione dei falsi positivi supera il leggero incremento dei falsi negativi.

Questi risultati evidenziano come la combinazione del parametro α e la scelta delle feature influiscano in maniera determinante sulla qualità operativa del modello, 
permettendo di ottenere un compromesso ottimale tra riduzione dei falsi positivi e mantenimento delle metriche di performance.
'''