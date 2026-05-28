import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import QAOAAnsatz

# EVOVAQ
from evovaq.problem import Problem
from evovaq.GeneticAlgorithm import GA
from evovaq.HillClimbing import HC
from evovaq.MemeticAlgorithm import MA
import evovaq.tools.operators as op

import json

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


# ------------------------------------------------------------
# LINEAR TERMS
# ------------------------------------------------------------
for _, row in df_single.iterrows():

    i = index[row["Sensor(s)"]]

    add(Q_base, i, i, row["Delta_L_mean"])


Li = {
    row["Sensor(s)"]: row["Delta_L_mean"]
    for _, row in df_single.iterrows()
}


# ------------------------------------------------------------
# PAIRWISE TERMS
# ------------------------------------------------------------
for _, row in df_pair.iterrows():

    s1, s2 = re.findall(r"PS\d+", row["Sensor(s)"])

    i, j = index[s1], index[s2]

    qij = row["Delta_L_mean"] - Li[s1] - Li[s2]

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

    return SparsePauliOp.from_list(list(pauli.items()))


# ============================================================
# PARAMETERS
# ============================================================
lambda_penalty = 1000

k_values = range(1, n + 1)

n_seeds = 10

seed_list = [42 * (i + 1) for i in range(n_seeds)]

print("Seeds:", seed_list)

regime = "real_statevector"

pareto_storage = {
    k: []
    for k in k_values
}

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
        # lambda (sum x_i - k)^2
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
        ).decompose()


        # ----------------------------------------------------
        # COST FUNCTION
        # ----------------------------------------------------
        def cost(params):

            qc = qaoa.assign_parameters(params)

            state = Statevector.from_instruction(qc)

            return np.real(
                state.expectation_value(H)
            )


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

            x_new[i] = np.random.uniform(low, high)

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
            max_gen=100,
            verbose=False,
            seed=seed
        )


        # ====================================================
        # STORE ENERGY
        # ====================================================
        energies_seed.append(res.fun)


        # ====================================================
        # FINAL STATE
        # ====================================================
        best_params = res.x

        qc_best = qaoa.assign_parameters(best_params)

        state_best = Statevector.from_instruction(qc_best)

        probs = state_best.probabilities_dict()


        # ----------------------------------------------------
        # MOST PROBABLE BITSTRING
        # ----------------------------------------------------
        best_bitstring = max(
            probs,
            key=probs.get
        )

        # reverse qiskit ordering
        best_bitstring = best_bitstring[::-1]

        solution = np.array(
            [int(b) for b in best_bitstring]
        )

        solutions_seed.append(solution)
        pareto_storage[k].append({
            "energy": float(res.fun),
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
mean_energies = np.mean(all_energies, axis=0)

std_energies = np.std(all_energies, axis=0)

mean_solutions = np.mean(all_solutions, axis=0)


# ============================================================
# PARETO SUMMARY SAVE
# ============================================================
pareto_summary = {
    regime: {}
}

for k in k_values:

    entries = pareto_storage[k]

    energies = np.array([
        e["energy"] for e in entries
    ])

    sols = np.array([
        e["solution"] for e in entries
    ])

    mean_energy = float(np.mean(energies))

    mean_solution = sols.mean(axis=0)

    mean_subset = (
        mean_solution > 0.5
    ).astype(int)

    selected_sensors_mean = [
        sensors[i]
        for i in range(n)
        if mean_subset[i] == 1
    ]

    pareto_summary[regime][k] = {

        "mean_energy": mean_energy,

        "std_energy": float(
            np.std(energies)
        ),

        "best_energy": float(
            np.min(energies)
        ),

        "mean_solution": mean_solution.tolist(),

        "selected_sensors_mean":
            selected_sensors_mean,

        "frequency":
            mean_solution.tolist()
    }


with open(
    f"pareto_summary_{regime}_gestup.json",
    "w"
) as f:

    json.dump(
        pareto_summary,
        f,
        indent=2
    )


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

# ------------------------------------------------------------
# X AXIS = SENSOR NAMES
# ------------------------------------------------------------
plt.xticks(list(k_values))

plt.xlabel("Number of selected sensors (k)")
plt.ylabel("Mean Energy")
plt.title("Energy vs Sensor Budget (Pareto-like Curve)")
plt.grid()
plt.tight_layout()

plt.savefig(
    "pareto_frontier_multiseed_gestup.pdf",
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
    origin="lower"   # k crescente verso l'alto
)

# ------------------------------------------------------------
# X AXIS = SENSOR NAMES
# ------------------------------------------------------------
plt.xticks(
    ticks=np.arange(n),
    labels=sensors,
    rotation=90
)

# ------------------------------------------------------------
# Y AXIS = REAL k VALUES
# ------------------------------------------------------------
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
    "sensor_selection_heatmap_multiseed_gestup.pdf",
    bbox_inches="tight"
)

plt.show()'''









