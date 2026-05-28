import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import json

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz

# Runtime primitives
from qiskit_ibm_runtime import Session
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler

# Transpilation
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Aer backend
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import (
    FakeWashingtonV2,
    FakeAlmadenV2,
    FakeTorontoV2,
    FakeCairoV2,
    FakeBrooklynV2,
    FakeCambridgeV2,
    FakeSingaporeV2
)

# EVOVAQ
from evovaq.problem import Problem
from evovaq.GeneticAlgorithm import GA
from evovaq.HillClimbing import HC
from evovaq.MemeticAlgorithm import MA
import evovaq.tools.operators as op



# ============================================================
# BACKEND
# ============================================================
def define_backend(use_noise=False):

    if use_noise:

        backend_fake = FakeWashingtonV2()

        noise_model = NoiseModel.from_backend(
            backend_fake
        )

        gpu_instance = AerSimulator(
            noise_model=noise_model,
            method="statevector",
            device="GPU"
        )

        gpu_instance.set_options(
            precision='single'
        )

        return noise_model, gpu_instance

    else:

        gpu_instance = AerSimulator(
            method="statevector",
            device="GPU"
        )

        gpu_instance.set_options(
            precision='single'
        )

        return gpu_instance


# ============================================================
# SELECT BACKEND
# ============================================================
_,backend = define_backend(use_noise=True)
#backend = define_backend(use_noise=False)

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv("qubo_sensor_statistics_gestup.csv")

df_single = df[~df["Sensor(s)"].str.startswith("(")].copy()
df_pair   = df[df["Sensor(s)"].str.startswith("(")].copy()

sensors = df_single["Sensor(s)"].tolist()

index = {s: i for i, s in enumerate(sensors)}

n = len(sensors)

print("Qubits:", n)


# ============================================================
# BUILD BASE QUBO
# ============================================================
Q_base = {}

def add(Q, i, j, v):

    if i > j:
        i, j = j, i

    Q[(i, j)] = Q.get((i, j), 0) + v


# ============================================================
# LINEAR TERMS
# ============================================================
for _, row in df_single.iterrows():

    i = index[row["Sensor(s)"]]

    add(
        Q_base,
        i,
        i,
        row["Delta_L_mean"]
    )


Li = {
    row["Sensor(s)"]: row["Delta_L_mean"]
    for _, row in df_single.iterrows()
}


# ============================================================
# PAIRWISE TERMS
# ============================================================
for _, row in df_pair.iterrows():

    s1, s2 = re.findall(
        r"PS\d+",
        row["Sensor(s)"]
    )

    i, j = index[s1], index[s2]

    qij = (
        row["Delta_L_mean"]
        - Li[s1]
        - Li[s2]
    )

    add(Q_base, i, j, qij)


# ============================================================
# QUBO → PAULI
# ============================================================
def qubo_to_pauli(Q, n):

    pauli = {}

    const = 0

    def add_p(p, c):

        pauli[p] = pauli.get(p, 0) + c

    for (i, j), c in Q.items():

        if i == j:

            z = ["I"] * n

            z[i] = "Z"

            add_p("".join(z), -c / 2)

            const += c / 2

        else:

            zi = ["I"] * n
            zi[i] = "Z"

            zj = ["I"] * n
            zj[j] = "Z"

            zij = ["I"] * n

            zij[i] = "Z"
            zij[j] = "Z"

            add_p("".join(zi), -c / 4)

            add_p("".join(zj), -c / 4)

            add_p("".join(zij), c / 4)

            const += c / 4

    add_p("I" * n, const)

    return SparsePauliOp.from_list(
        list(pauli.items())
    )


# ============================================================
# CX SCALING ANALYSIS
# ============================================================
def experiment_cx_scaling_sensor_selection(
    Q,
    n,
    p_values,
    save_dir="./cx_scaling/"
):

    os.makedirs(save_dir, exist_ok=True)


    # ========================================================
    # BACKENDS
    # ========================================================
    backends = {

        "FakeWashingtonV2": FakeWashingtonV2(),

        "FakeSingaporeV2": FakeSingaporeV2(),

        "FakeCambridgeV2": FakeCambridgeV2(),

        "FakeBrooklynV2": FakeBrooklynV2(),

        "FakeAlmadenV2": FakeAlmadenV2(),

        "FakeTorontoV2": FakeTorontoV2(),

        "FakeCairoV2": FakeCairoV2()
    }


    # ========================================================
    # STORAGE
    # ========================================================
    cx_before = []

    cx_after = {
        name: []
        for name in backends
    }


    # ========================================================
    # HAMILTONIAN
    # ========================================================
    H = qubo_to_pauli(Q, n)


    # ========================================================
    # LOOP OVER p
    # ========================================================
    for i, p in enumerate(p_values):

        print(f"Processing p = {p}")


        # ----------------------------------------------------
        # BUILD QAOA
        # ----------------------------------------------------
        qaoa = QAOAAnsatz(
            cost_operator=H,
            reps=p
        )

        qaoa.measure_all()


        # ----------------------------------------------------
        # BEFORE TRANSPILATION
        # ----------------------------------------------------
        cx_count_before = (
            qaoa
            .decompose(reps=10)
            .count_ops()
            .get("cx", 0)
        )

        cx_before.append(cx_count_before)


        # ----------------------------------------------------
        # SAVE RAW CIRCUIT
        # ----------------------------------------------------
        if i == 0:

            qaoa.decompose(reps=10).draw(
                output="mpl"
            ).savefig(
                os.path.join(
                    save_dir,
                    "qaoa_before_transpilation.pdf"
                )
            )


        # ====================================================
        # TRANSPILATION FOR EACH BACKEND
        # ====================================================
        for name, backend in backends.items():

            pm = generate_preset_pass_manager(
                optimization_level=3,
                backend=backend,
                seed_transpiler=42
            )

            transpiled = pm.run(qaoa)


            # ------------------------------------------------
            # COUNT CX
            # ------------------------------------------------
            cx_count_after = (
                transpiled
                .decompose(reps=10)
                .count_ops()
                .get("cx", 0)
            )

            cx_after[name].append(
                cx_count_after
            )


            # ------------------------------------------------
            # SAVE EXAMPLE TRANSPILED CIRCUIT
            # ------------------------------------------------
            if i == 0 and name == "FakeWashingtonV2":

                transpiled.decompose(reps=10).draw(
                    output="mpl"
                ).savefig(
                    os.path.join(
                        save_dir,
                        "qaoa_after_transpilation_fakewashington.pdf"
                    )
                )


    # ========================================================
    # PLOT
    # ========================================================
    plt.figure(figsize=(11, 6))


    # --------------------------------------------------------
    # BEFORE
    # --------------------------------------------------------
    plt.plot(
        p_values,
        cx_before,
        marker="x",
        linewidth=3,
        markersize=10,
        linestyle="-",
        color="black",
        label="Before transpilation"
    )


    # --------------------------------------------------------
    # AFTER
    # --------------------------------------------------------
    for name, counts in cx_after.items():

        plt.plot(
            p_values,
            counts,
            marker="o",
            linewidth=2.5,
            linestyle="-",
            label=name
        )


    # ========================================================
    # STYLE
    # ========================================================
    plt.xlabel("QAOA depth p")

    plt.ylabel("Number of CX gates")

    plt.title("CX Scaling Before/After Transpilation")

    plt.xticks(p_values)

    plt.grid(
        True,
        linestyle="--",
        alpha=0.6
    )

    plt.legend()

    plt.tight_layout()


    # ========================================================
    # SAVE
    # ========================================================
    plt.savefig(
        os.path.join(
            save_dir,
            "cx_scaling_comparison.pdf"
        )
    )

    plt.close()


    return cx_before, cx_after




# ============================================================
# PARAMETERS
# ============================================================
lambda_penalty = 1000

k_values = range(1, n + 1)

regime = "noisyshots_2048"

n_seeds = 3 #10

seed_list = [42 * (i + 1) for i in range(n_seeds)]

print("Seeds:", seed_list)

pareto_storage = {
    k: []
    for k in k_values
}



# ============================================================
# CX SCALING ANALYSIS
# ============================================================

# scegli un k rappresentativo
'''k_cx = 12

Q_cx = Q_base.copy()

# cardinality penalty
for i in range(n):

    Q_cx[(i, i)] = (
        Q_cx.get((i, i), 0)
        + lambda_penalty * (1 - 2 * k_cx)
    )

for i in range(n):
    for j in range(i + 1, n):

        Q_cx[(i, j)] = (
            Q_cx.get((i, j), 0)
            + 2 * lambda_penalty
        )


# profondità QAOA da testare
p_values_cx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


experiment_cx_scaling_sensor_selection(
    Q=Q_cx,
    n=n,
    p_values=p_values_cx,
    save_dir="./cx_scaling_gestup/"
)'''


# ============================================================
# GLOBAL STORAGE
# ============================================================
all_energies = []
all_solutions = []


# ============================================================
# MAIN LOOP OVER SEEDS
# ============================================================
for seed in seed_list:

    print(f"\n================ SEED {seed} ================\n")

    np.random.seed(seed)

    energies_seed = []

    solutions_seed = []


    # ========================================================
    # LOOP OVER k
    # ========================================================
    for k in k_values:

        print(f"Seed {seed} | k = {k}")

        # ----------------------------------------------------
        # COPY BASE QUBO
        # ----------------------------------------------------
        Q = Q_base.copy()


        # ----------------------------------------------------
        # CARDINALITY CONSTRAINT
        # ----------------------------------------------------
        for i in range(n):

            Q[(i, i)] = (
                Q.get((i, i), 0)
                + lambda_penalty * (1 - 2 * k)
            )

        for i in range(n):
            for j in range(i + 1, n):

                Q[(i, j)] = (
                    Q.get((i, j), 0)
                    + 2 * lambda_penalty
                )


        # ----------------------------------------------------
        # HAMILTONIAN
        # ----------------------------------------------------
        H = qubo_to_pauli(Q, n)


        # ----------------------------------------------------
        # QAOA
        # ----------------------------------------------------
        qaoa = QAOAAnsatz(
            cost_operator=H,
            reps=4
        )

        qaoa.measure_all()


        # ----------------------------------------------------
        # TRANSPILATION
        # ----------------------------------------------------
        pm = generate_preset_pass_manager(
            optimization_level=3,
            backend=backend,
            seed_transpiler=42
        )

        qaoa = pm.run(qaoa)


        # ----------------------------------------------------
        # COST FUNCTION VIA ESTIMATOR
        # ----------------------------------------------------
        def cost(params):

            qc = qaoa.assign_parameters(params)

            with Session(backend=backend) as session:

                estimator = Estimator(
                    mode=session
                )

                # deterministic expectation value
                estimator.options.default_shots = 2048

                job = estimator.run([
                    (qc, H)
                ])

                result = job.result()

                energy = result[0].data.evs

            return np.real(energy)


        # ----------------------------------------------------
        # BOUNDS
        # ----------------------------------------------------
        bounds = [
            (-np.pi, np.pi)
            for _ in range(qaoa.num_parameters)
        ]


        # ----------------------------------------------------
        # EVOVAQ PROBLEM
        # ----------------------------------------------------
        problem = Problem(
            qaoa.num_parameters,
            bounds,
            lambda x: cost(x)
        )


        # ====================================================
        # GENETIC ALGORITHM
        # ====================================================
        global_search = GA(
            selection=op.sel_tournament,
            crossover=op.cx_uniform,
            mutation=op.mut_gaussian,
            sigma=0.2,
            mut_indpb=0.15,
            cxpb=0.9,
            tournsize=5
        )


        # ====================================================
        # LOCAL SEARCH
        # ====================================================
        def get_neighbour(problem, x):

            x_new = x.copy()

            i = np.random.randint(len(x))

            low, high = problem.param_bounds[0]

            x_new[i] = np.random.uniform(
                low,
                high
            )

            return x_new


        local_search = HC(
            generate_neighbour=get_neighbour
        )


        # ====================================================
        # MEMETIC ALGORITHM
        # ====================================================
        memetic = MA(
            global_search=global_search.evolve_population,
            sel_for_refinement=op.sel_best,
            local_search=local_search.stochastic_var,
            frequency=0.1,
            intensity=5
        )


        # ====================================================
        # OPTIMIZATION
        # ====================================================
        res = memetic.optimize(
            problem,
            10,
            max_gen=15, #100
            verbose=False,
            seed=seed
        )


        # ====================================================
        # STORE ENERGY
        # ====================================================
        energies_seed.append(res.fun)


        # ====================================================
        # FINAL CIRCUIT
        # ====================================================
        best_params = res.x

        qc_best = qaoa.assign_parameters(
            best_params
        )


        # ====================================================
        # SAMPLER
        # ====================================================
        with Session(backend=backend) as session:

            sampler = Sampler(
                mode=session
            )

            # deterministic probabilities
            sampler.options.default_shots = 2048

            job = sampler.run([
                qc_best
            ])

            counts = (
                job.result()[0]
                .data
                .meas
                .get_counts()
            )


        # ----------------------------------------------------
        # MOST PROBABLE BITSTRING
        # ----------------------------------------------------
        best_bitstring = max(
            counts,
            key=counts.get
        )

        # reverse qiskit ordering
        best_bitstring = best_bitstring[::-1]

        solution = np.array(
            [int(b) for b in best_bitstring]
        )

        solutions_seed.append(solution)

        pareto_storage[k].append({
            "energy": res.fun,
            "solution": solution.copy()
        })


    # ========================================================
    # SAVE SEED RESULTS
    # ========================================================
    all_energies.append(energies_seed)

    all_solutions.append(solutions_seed)


# ============================================================
# CONVERT TO NUMPY
# ============================================================
all_energies = np.array(all_energies)

all_solutions = np.array(all_solutions)

print("\nall_energies shape:", all_energies.shape)

print("all_solutions shape:", all_solutions.shape)


# ============================================================
# AVERAGE OVER SEEDS
# ============================================================
mean_energies = np.mean(
    all_energies,
    axis=0
)

std_energies = np.std(
    all_energies,
    axis=0
)

mean_solutions = np.mean(
    all_solutions,
    axis=0
)



pareto_summary = {
    regime: {}
}

for k in k_values:

    entries = pareto_storage[k]

    energies = np.array([e["energy"] for e in entries])
    sols = np.array([e["solution"] for e in entries])

    mean_energy = float(np.mean(energies))

    mean_solution = sols.mean(axis=0)
    mean_subset = (mean_solution > 0.5).astype(int)

    selected_sensors_mean = [
        sensors[i] for i in range(n) if mean_subset[i] == 1
    ]

    pareto_summary[regime][k] = {
        "mean_energy": mean_energy,
        "std_energy": float(np.std(energies)),
        "best_energy": float(np.min(energies)),
        "mean_solution": mean_solution.tolist(),
        "selected_sensors_mean": selected_sensors_mean,
        "frequency": sols.mean(axis=0).tolist()
    }

with open(f"pareto_summary_{regime}_gestup.json", "w") as f:
    json.dump(pareto_summary, f, indent=2)


# ============================================================
# PARETO FRONTIER
# ============================================================
'''plt.figure(figsize=(10, 5))

plt.errorbar(
    list(k_values),
    mean_energies,
    yerr=std_energies,
    marker="o",
    capsize=5
)

plt.xticks(list(k_values))
plt.xlabel("Number of selected sensors (k)")
plt.ylabel("Mean energy")
plt.title("Energy vs Sensor Budget (Pareto curve)")

plt.grid()

plt.tight_layout()

plt.savefig(
    "pareto_frontier_multiseed_gestup_real_sim_statevector.pdf",
    bbox_inches="tight"
)

plt.show()'''


# ============================================================
# HEATMAP
# ============================================================
'''plt.figure(figsize=(12, 6))

im = plt.imshow(
    mean_solutions,
    aspect="auto",
    cmap="Blues",
    vmin=0,
    vmax=1,
    origin="lower"
)

plt.xticks(
    ticks=np.arange(n),
    labels=sensors,
    rotation=90
)

plt.yticks(
    ticks=np.arange(len(k_values)),
    labels=list(k_values)
)

plt.xlabel("Sensors")

plt.ylabel("k")

plt.title("Sensor Selection Frequency Across Seeds")

plt.colorbar(
    im,
    label="Selection Frequency"
)

plt.tight_layout()

plt.savefig(
    "sensor_selection_heatmap_multiseed_gestup_real_sim_statevector.pdf",
    bbox_inches="tight"
)

plt.show()'''