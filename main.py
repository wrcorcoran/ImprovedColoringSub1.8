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

eta = {i: m.addVar(lb=0.0, ub=1, name=f"eta_{i}") for i in range(1, DC_MAX + 1)}
P = {i: m.addVar(lb=0.0, ub=1.0, name=f"P_{i}") for i in range(1, N_MAX + 1)}
lam = {cfg: m.addVar(name=f"lambda_{cfg}") for cfg in configs}


# scaled etas (η_i)
def Eta(i: int):
    return eta[i]


def Gamma(cfg):
    if cfg in UNBLOCKED_CONFIGS:
        return lmbda - (1 - 2 * P[2])
    return lmbda - (1 - 2 * P[3])


def lambda_bounds(cfg):
    bounds = []
    if cfg == (2, 2):
        if USE_ORIGINAL:
            alpha_1_val = 0
        else:
            alpha_1_val = Eta(1) * (Gamma(cfg) + (1 + P[2]))
        bounds.append(2 - P[2] + alpha_1_val)

    elif cfg == (3, 3):
        if USE_ORIGINAL:
            bounds.append(1.5 + 0.5 * P[3])
        else:
            expr1 = (
                1.5
                + 0.5 * P[3]
                + Eta(2) * Gamma(cfg)
                + 2 * Eta(2)
                - (1 - P[3]) * Eta(1)
                + (Eta(2) - Eta(1)) * 2 * lmbda
            )
            expr2 = (
                1.5
                + 0.5 * P[3]
                + Eta(2) * Gamma(cfg)
                + (1 + P[2]) * Eta(2)
                + (Eta(2) - Eta(1)) * 2 * lmbda
            )
            bounds.extend([expr1, expr2])

    elif cfg == (3, 2):
        if USE_ORIGINAL:
            beta_1_val = 0
        else:
            beta_1_val = Eta(1) * (lmbda - 1)
        bounds.append(2 - P[3] - beta_1_val)

    elif cfg == (5, 3):
        if USE_ORIGINAL:
            beta_2_val = 0
        else:
            beta_2_val = Eta(2) * (lmbda - 1)
        bounds.append(1.5 + P[2] + 0.5 * P[5] - beta_2_val)

    elif cfg == (4, 2):
        bounds.append(2 - P[2] + 2 * (P[3] - P[4]))

    elif cfg == (7, 3):
        bounds.append(1.5 + 2 * P[3])

    else:
        raise ValueError(f"Unknown config: {cfg}")

    return bounds


# Core constraints
m.setObjective(lmbda, GRB.MINIMIZE)

# P_1 = 1
m.addConstr(P[1] == 1, name="P1_eq_1")

for j in range(2, N_MAX + 1):
    m.addConstr(P[j] <= (2.0 / 3.0) * P[j - 1], name=f"P_decay_{j}")

for i in range(1, DC_MAX):
    m.addConstr(eta[i] <= eta[i + 1], name=f"eta_monotone_{i}")

# lambda constraints
for cfg in configs:
    lbs = lambda_bounds(cfg)
    for k, expr in enumerate(lbs):
        m.addConstr(lam[cfg] >= expr, name=f"lambda_{cfg}_{k}")
    m.addConstr(lmbda >= lam[cfg], name=f"obj_bounds_{cfg}")

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


def lambda_bounds_numeric(cfg, fixed):
    # fixed: dict with keys "lambda", "eta", "P"
    lmbda_val = fixed["lambda"]
    eta_val = fixed["eta"]
    P_val = fixed["P"]
    bounds = []
    if cfg == (2, 2):
        if USE_ORIGINAL:
            alpha_1_val = 0
        else:
            alpha_1_val = eta_val[1] * (lmbda_val - (1 - 2 * P_val[2]) + (1 + P_val[2]))
        bounds.append(2 - P_val[2] + alpha_1_val)

    elif cfg == (3, 3):
        if USE_ORIGINAL:
            bounds.append(1.5 + 0.5 * P_val[3])
        else:
            gamma = lmbda_val - (1 - 2 * P_val[3])
            expr1 = (
                1.5
                + 0.5 * P_val[3]
                + eta_val[2] * gamma
                + 2 * eta_val[2]
                - (1 - P_val[3]) * eta_val[1]
                + (eta_val[2] - eta_val[1]) * 2 * lmbda_val
            )
            expr2 = (
                1.5
                + 0.5 * P_val[3]
                + eta_val[2] * gamma
                + (1 + P_val[2]) * eta_val[2]
                + (eta_val[2] - eta_val[1]) * 2 * lmbda_val
            )
            bounds.extend([expr1, expr2])

    elif cfg == (3, 2):
        if USE_ORIGINAL:
            beta_1_val = 0
        else:
            beta_1_val = eta_val[1] * (lmbda_val - 1)
        bounds.append(2 - P_val[3] - beta_1_val)

    elif cfg == (5, 3):
        if USE_ORIGINAL:
            beta_2_val = 0
        else:
            if (5, 3) in DC_EQ1_CASES:
                beta_2_val = eta_val[1] * (lmbda_val - 1)
            else:
                beta_2_val = eta_val[2] * (lmbda_val - 1)
        bounds.append(1.5 + P_val[2] + 0.5 * P_val[5] - beta_2_val)

    elif cfg == (4, 2):
        bounds.append(2 - P_val[2] + 2 * (P_val[3] - P_val[4]))

    elif cfg == (7, 3):
        bounds.append(1.5 + 2 * P_val[3])

    else:
        raise ValueError(f"Unknown config: {cfg}")
    return bounds


def print_lambda_calculations(fixed_vals=None):
    print("\n--- Lambda configuration values (recomputed) ---")
    if fixed_vals is not None:
        for cfg in configs:
            try:
                vals = lambda_bounds_numeric(cfg, fixed_vals)
                for v in vals:
                    print(f"lambda_calc{cfg}: {v:.9g}")
            except Exception as e:
                print(f"lambda_calc{cfg}: ERROR: {e}")
    else:
        for cfg in configs:
            for val in lambda_bounds(cfg):
                try:
                    print(f"lambda_calc{cfg}: {val.getValue():.9g}")
                except Exception as e:
                    print(f"lambda_calc{cfg}: ERROR: {e}")


def run_fix_all():
    # Collect numeric values for lambda, eta, P
    fixed_lambda = None
    fixed_eta = {}
    fixed_P = {}
    # Prompt for lambda
    try:
        s = input(f"Fix lambda? Enter value or press Enter to skip: ").strip()
    except EOFError:
        s = ""
    if s != "":
        try:
            fixed_lambda = float(s)
        except ValueError:
            raise ValueError(f"Invalid numeric value for lambda: {s}")
    # Prompt for all eta[i]
    for i in sorted(eta.keys()):
        try:
            s = input(f"Fix eta_{i}? Enter value or press Enter to skip: ").strip()
        except EOFError:
            s = ""
        if s != "":
            try:
                fixed_eta[i] = float(s)
            except ValueError:
                raise ValueError(f"Invalid numeric value for eta_{i}: {s}")
    # Prompt for all P[i]
    for i in sorted(P.keys()):
        try:
            s = input(f"Fix P_{i}? Enter value or press Enter to skip: ").strip()
        except EOFError:
            s = ""
        if s != "":
            try:
                fixed_P[i] = float(s)
            except ValueError:
                raise ValueError(f"Invalid numeric value for P_{i}: {s}")
    # Fill missing eta/P with zeros (or leave as is? We'll require all values)
    # To make sure all needed values are present, fill missing with zeros
    for i in eta.keys():
        if i not in fixed_eta:
            fixed_eta[i] = 0.0
    for i in P.keys():
        if i not in fixed_P:
            fixed_P[i] = 0.0
    if fixed_lambda is None:
        fixed_lambda = 0.0
    return {"lambda": fixed_lambda, "eta": fixed_eta, "P": fixed_P}


def choose_mode_and_solve():
    print("Mode options:")
    print("  1) Solve without fixing any variables")
    print("  2) Interactively choose variables to fix (press Enter to skip each)")
    print(
        "  3) Fix all variables (including lambda) and only print lambda configuration values"
    )
    try:
        mode = input("Select mode [1/2/3]: ").strip()
    except EOFError:
        mode = "1"

    if mode == "2":
        run_interactive_fixing()
    elif mode == "3":
        fixed_vals = run_fix_all()
        print_lambda_calculations(fixed_vals)
        return

    m.optimize()

    if m.status == GRB.OPTIMAL:
        print("\nOptimal solution:")
        print(f"lambda: {(lmbda.X):.9g}")
        print("eta:")
        for i in sorted(eta.keys()):
            print(f"  eta_{i}: {eta[i].X:.9g}")
        print("P:")
        for i in sorted(P.keys()):
            print(f"  P_{i}: {P[i].X:.9g}")
        print("lambdas:")
        print_lambda_calculations()
    else:
        print(f"Model status: {m.status}")


if __name__ == "__main__":
    choose_mode_and_solve()
