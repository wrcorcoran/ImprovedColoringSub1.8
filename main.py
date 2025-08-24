import gurobipy as gp
from gurobipy import GRB
import argparse

# parse arguments
USE_ORIGINAL = False
parser = argparse.ArgumentParser()
parser.add_argument(
    "--original",
    action="store_true",
    help="If set, alpha/beta functions return 0, should recreate 1.83333",
)
args = parser.parse_args()
USE_ORIGINAL = args.original

# Data
N_MAX = 7
DC_MAX = 2
Delta = 1000
configs = [(2, 2), (3, 3), (3, 2), (5, 3), (4, 2), (7, 3)]

UNBLOCKED_CONFIGS = {(2, 2), (3, 3)}
DC_EQ1_CASES = {(2, 2), (3, 2), (4, 2)}

# Model
m = gp.Model("colorings_qp")

# Variables
lmbda = m.addVar(lb=0.0, name="lambda")

alpha_1 = m.addVar(name="alpha_1")
alpha_2 = m.addVar(name="alpha_2")
beta_1 = m.addVar(name="beta_1")
beta_2 = m.addVar(name="beta_2")

eta = {i: m.addVar(lb=0.0, ub=1.0, name=f"eta_{i}") for i in range(1, DC_MAX + 1)}

P = {i: m.addVar(lb=0.0, ub=1.0, name=f"P_{i}") for i in range(1, N_MAX + 1)}

lam = {cfg: m.addVar(name=f"lambda_{cfg}") for cfg in configs}


# scaled etas (η_i / Δ)
def Eta(i: int):
    return eta[i] / Delta


def Gamma(cfg):
    if cfg in UNBLOCKED_CONFIGS:
        return lmbda - (1 - 2 * P[2]) * Delta
    return lmbda - (1 - 2 * P[3]) * Delta


def Alpha1(cfg):
    if USE_ORIGINAL:
        return gp.LinExpr(0)
    return Eta(1) * (Gamma(cfg) + (1 + P[2]) * Delta)


def Alpha2_case1(cfg):
    if USE_ORIGINAL:
        return gp.LinExpr(0)
    return (
        Eta(2) * Gamma(cfg)
        + 2 * Eta(2)
        - (1 - P[3]) * Delta * Eta(1)
        + (Eta(2) - Eta(1)) * 2 * lmbda
    )


def Alpha2_case2(cfg):
    if USE_ORIGINAL:
        return gp.LinExpr(0)
    return Eta(2) * Gamma(cfg) + (1 + P[2]) * Eta(2) + (Eta(2) - Eta(1)) * 2 * lmbda


def Beta(cfg):
    if USE_ORIGINAL:
        return gp.LinExpr(0)
    if cfg in DC_EQ1_CASES:
        return Eta(1) * (lmbda - Delta - 2)
    return Eta(2) * (lmbda - Delta - 2)


# Core constraints
# Objective: minimize the worst-case lambda
m.setObjective(lmbda, GRB.MINIMIZE)

# P_1 = 1
m.addConstr(P[1] == 1, name="P1_eq_1")

for j in range(2, N_MAX + 1):
    m.addConstr(P[j] <= (2.0 / 3.0) * P[j - 1], name=f"P_decay_{j}")

for i in range(1, DC_MAX):
    m.addConstr(eta[i] <= eta[i + 1], name=f"eta_monotone_{i}")

cfg = (2, 2)
m.addConstr(alpha_1 >= Alpha1(cfg), name="alpha1_bound")
m.addConstr(lam[cfg] >= 2 - P[2] + alpha_1, name="lambda_22")

cfg = (3, 3)
m.addConstr(alpha_2 >= Alpha2_case1(cfg), name="alpha2_case1")
m.addConstr(alpha_2 >= Alpha2_case2(cfg), name="alpha2_case2")
m.addConstr(lam[cfg] >= 1.5 + 0.5 * P[3] + alpha_2, name="lambda_33")

cfg = (3, 2)
m.addConstr(beta_1 == Beta(cfg), name="beta1_def")
m.addConstr(lam[cfg] >= 2 - P[3] - beta_1, name="lambda_32")

cfg = (5, 3)
m.addConstr(beta_2 == Beta(cfg), name="beta2_def")
m.addConstr(lam[cfg] >= 1.5 + P[2] + 0.5 * P[5] - beta_2, name="lambda_53")

cfg = (4, 2)
m.addConstr(lam[cfg] >= 2 - P[2] + 2 * (P[3] - P[4]), name="lambda_42")

cfg = (7, 3)
m.addConstr(lam[cfg] >= 1.5 + 2 * P[3], name="lambda_73")

for cfg in configs:
    m.addConstr(lmbda >= lam[cfg] * Delta, name=f"obj_bounds_{cfg}")


def prompt_fix(var, name):
    try:
        s = input(f"Fix {name}? Enter value or press Enter to skip: ").strip()
    except EOFError:
        s = ""
    if s == "":
        return  # no fixing
    try:
        val = float(s)
    except ValueError:
        raise ValueError(f"Invalid numeric value for {name}: {s}")
    m.addConstr(var == val, name=f"fix_{name}")


def run_interactive_fixing():
    prompt_fix(lmbda, "lmbda")
    prompt_fix(alpha_1, "alpha_1")
    prompt_fix(alpha_2, "alpha_2")
    prompt_fix(beta_1, "beta_1")
    prompt_fix(beta_2, "beta_2")

    for i in sorted(eta.keys()):
        prompt_fix(eta[i], f"eta_{i}")

    for i in sorted(P.keys()):
        prompt_fix(P[i], f"P_{i}")

    for cfg in configs:
        prompt_fix(lam[cfg], f"lambda_{cfg}")

    prompt_fix(lmbda, "lambda")


def print_lambda_calculations():
    """After solving, compute each lambda expression from the definitions."""
    if m.status != GRB.OPTIMAL:
        print("Model not solved to optimality.")
        return

    vals = {
        "lambda": lmbda.X,
        "alpha_1": alpha_1.X,
        "alpha_2": alpha_2.X,
        "beta_1": beta_1.X,
        "beta_2": beta_2.X,
    }
    vals.update({f"eta_{i}": eta[i].X for i in eta})
    vals.update({f"P_{i}": P[i].X for i in P})

    print("\n--- Lambda configuration values (recomputed) ---")
    for cfg in configs:
        if cfg == (2, 2):
            val = 2 - P[2].X + Alpha1(cfg).getValue()
        elif cfg == (3, 3):
            val = (
                1.5
                + 0.5 * P[3].X
                + min(Alpha2_case1(cfg).getValue(), Alpha2_case2(cfg).getValue())
            )
        elif cfg == (3, 2):
            val = 2 - P[3].X - Beta(cfg).getValue()
        elif cfg == (5, 3):
            val = 1.5 + P[2].X + 0.5 * P[5].X - Beta(cfg).getValue()
        elif cfg == (4, 2):
            val = 2 - P[2].X + 2 * (P[3].X - P[4].X)
        elif cfg == (7, 3):
            val = 1.5 + 2 * P[3].X
        else:
            val = float("nan")

        print(f"lambda_calc{cfg}: {val:.9g}")


def choose_mode_and_solve():
    print("Mode options:")
    print("  1) Solve without fixing any variables")
    print("  2) Interactively choose variables to fix (press Enter to skip each)")
    try:
        mode = input("Select mode [1/2]: ").strip()
    except EOFError:
        mode = "1"

    if mode == "2":
        run_interactive_fixing()

    m.optimize()

    if m.status == GRB.OPTIMAL:
        print("\nOptimal solution:")
        print(f"lambda: {(lmbda.X / Delta):.9g}")
        print("eta:")
        for i in sorted(eta.keys()):
            print(f"  eta_{i}: {eta[i].X:.9g}")
        print("P:")
        for i in sorted(P.keys()):
            print(f"  P_{i}: {P[i].X:.9g}")
        print("lambdas:")
        print_lambda_calculations()
        # for cfg in configs:
        #     print(f"  lambda_{cfg}: {lam[cfg].X:.9g}")
        print("alphas/betas:")
        print(f"  alpha_1: {alpha_1.X:.9g}")
        print(f"  alpha_2: {alpha_2.X:.9g}")
        print(f"  beta_1:  {beta_1.X:.9g}")
        print(f"  beta_2:  {beta_2.X:.9g}")
    else:
        print(f"Model status: {m.status}")


if __name__ == "__main__":
    choose_mode_and_solve()
