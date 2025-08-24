import gurobipy as gp
from gurobipy import GRB
import argparse

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--original",
    action="store_true",
    help="If set, alpha/beta functions return 0, should recreate 1.83333",
)
args = parser.parse_args()
USE_ORIGINAL = args.original

################################
## QUADRATIC PROGRAM BEGINS HERE
################################

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
alpha_2_case_1 = m.addVar(name="alpha_2_case_1")
alpha_2_case_2 = m.addVar(name="alpha_2_case_2")
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


def lambda_expression(cfg):
    if cfg == (2, 2):
        return 2 - P[2] + alpha_1
    elif cfg == (3, 3):
        return 1.5 + 0.5 * P[3] + alpha_2
    elif cfg == (3, 2):
        return 2 - P[3] - beta_1
    elif cfg == (5, 3):
        return 1.5 + P[2] + 0.5 * P[5] - beta_2
    elif cfg == (4, 2):
        return 2 - P[2] + 2 * (P[3] - P[4])
    elif cfg == (7, 3):
        return 1.5 + 2 * P[3]
    else:
        raise ValueError(f"Unknown config: {cfg}")


# Core constraints
m.setObjective(lmbda, GRB.MINIMIZE)

# P_1 = 1
m.addConstr(P[1] == 1, name="P1_eq_1")

for j in range(2, N_MAX + 1):
    m.addConstr(P[j] <= (2.0 / 3.0) * P[j - 1], name=f"P_decay_{j}")

for i in range(1, DC_MAX):
    m.addConstr(eta[i] <= eta[i + 1], name=f"eta_monotone_{i}")

# alpha/beta bounds
cfg = (2, 2)
m.addConstr(alpha_1 >= Alpha1(cfg), name="alpha1_bound")

# NOTE: I think there's a logic error here, because it's being flipped to >=
# the portion that should be a min becomes a max
cfg = (3, 3)
m.addConstr(alpha_2_case_1 >= Alpha2_case1(cfg), name="alpha2_case1")
m.addConstr(alpha_2_case_2 >= Alpha2_case2(cfg), name="alpha2_case2")
m.addConstr(alpha_2 >= alpha_2_case_1, name="alpha2_case1")
m.addConstr(alpha_2 >= alpha_2_case_2, name="alpha2_case2")

cfg = (3, 2)
m.addConstr(beta_1 == Beta(cfg), name="beta1_def")

cfg = (5, 3)
m.addConstr(beta_2 == Beta(cfg), name="beta2_def")

# lambda constraints
for cfg in configs:
    m.addConstr(lam[cfg] >= lambda_expression(cfg), name=f"lambda_{cfg}")
    m.addConstr(lmbda >= lam[cfg] * Delta, name=f"obj_bounds_{cfg}")

################################
## QUADRATIC PROGRAM ENDS HERE
################################


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
    for i in sorted(eta.keys()):
        prompt_fix(eta[i], f"eta_{i}")

    for i in sorted(P.keys()):
        prompt_fix(P[i], f"P_{i}")


def print_lambda_calculations():
    if m.status != GRB.OPTIMAL:
        print("Model not solved to optimality.")
        return
    print("\n--- Lambda configuration values (recomputed) ---")
    for cfg in configs:
        val = lambda_expression(cfg).getValue()
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
        print("alphas/betas:")
        print(f"  alpha_1: {alpha_1.X:.9g}")
        print(f"  alpha_2: {alpha_2.X:.9g}")
        print(f"  alpha_1_case_1: {alpha_2_case_1.X:.9g}")
        print(f"  alpha_2_case_2: {alpha_2_case_2.X:.9g}")
        print(f"  beta_1:  {beta_1.X:.9g}")
        print(f"  beta_2:  {beta_2.X:.9g}")
    else:
        print(f"Model status: {m.status}")


if __name__ == "__main__":
    choose_mode_and_solve()