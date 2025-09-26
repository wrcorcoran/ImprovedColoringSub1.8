import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import minimize, differential_evolution, shgo
import numpy as np

################################
## QUADRATIC PROGRAM BEGINS HERE
################################

# Data
N_MAX = 7
DC_MAX = 2
configs = [(2, 2), (3, 3), (3, 2), (5, 3), (4, 2), (7, 3)]

UNBLOCKED_CONFIGS = {(2, 2), (3, 3)}
DC_EQ1_CASES = {(2, 2), (3, 2), (4, 2)}

def Gamma(cfg, lmbda, P):
    return lmbda - (1 - P[cfg[0]] - P[cfg[1]])

def lambda_bounds(cfg, lmbda, P, eta):
    bounds = []
    if cfg == (2, 2):
        alpha_1_val = eta[1] * (Gamma(cfg, lmbda, P) + (1 + P[2]))
        bounds.append(2 - P[2] + alpha_1_val)

    elif cfg == (3, 3):
        expr1 = (
            1.5
            + 0.5 * P[3]
            + eta[2] * Gamma(cfg, lmbda, P)
            + 2 * eta[2]
            - (1 - P[3]) * eta[1]
            + (eta[2] - eta[1]) * (2 * lmbda)
        )
        expr2 = (
            1.5
            + 0.5 * P[3]
            + eta[2] * Gamma(cfg, lmbda, P)
            + (1 + P[2]) * eta[2]
            + (eta[2] - eta[1]) * (2 * lmbda)
        )
        bounds.extend([expr1, expr2])

    elif cfg == (3, 2):
        beta_1_val = eta[1] * (lmbda - 1)
        bounds.append(2 - P[3] - beta_1_val)

    elif cfg == (5, 3):
        beta_2_val = eta[2] * (lmbda - 1)
        bounds.append(1.5 + P[2] + 0.5 * P[5] - beta_2_val)

    elif cfg == (4, 2):
        bounds.append(2 - P[2] + 2 * (P[3] - P[4]))

    elif cfg == (7, 3):
        bounds.append(1.5 + 2 * P[3])

    return bounds

def build_model_for_eta(eta_vals):
    m = gp.Model("colorings_qp")
    m.Params.OutputFlag = 0

    # Variables
    lmbda = m.addVar(lb=0.0, ub=2.0, name="lambda")
    P = {i: m.addVar(lb=0.0, ub=1.0, name=f"P_{i}") for i in range(1, N_MAX + 1)}

    # Constraints
    m.setObjective(lmbda, GRB.MINIMIZE)
    m.addConstr(P[1] == 1, name="P1_eq_1")

    for j in range(3, N_MAX + 1):
        m.addConstr(P[j] <= (2.0 / 3.0) * P[j - 1], name=f"P_decay_{j}")

    # lambda constraints
    for cfg in configs:
        lbs = lambda_bounds(cfg, lmbda, P, eta_vals)
        for k, expr in enumerate(lbs):
            m.addConstr(lmbda >= expr, name=f"lambda_{cfg}_{k}")

    return m, lmbda, P

def solve_model_return(m, lmbda_var, P_vars):
    m.optimize()
    if m.status == GRB.OPTIMAL:
        lmbda_val = lmbda_var.X
        P_val = {i: P_vars[i].X for i in P_vars}
        return lmbda_val, P_val
    else:
        return None, None

################################
## OPTIMIZATION OVER ETA
################################

def objective(eta_vec):
    eta_vals = {1: eta_vec[0], 2: eta_vec[1]}
    if eta_vals[2] < eta_vals[1]:
        return 1e6
    m, lmbda_var, P_vars = build_model_for_eta(eta_vals)
    lmbda_val, _ = solve_model_return(m, lmbda_var, P_vars)
    if lmbda_val is None:
        return 1e6 
    return lmbda_val

def optimize_eta():
    bounds = [(0.0, 0.1), (0.0, 0.1)]
    x0 = [0.0565767814, 0.0710285694]
    res = minimize(objective, x0, bounds=bounds, method="Powell")
    best_eta = {1: res.x[0], 2: res.x[1]}
    best_lambda = res.fun
    return best_eta, best_lambda

if __name__ == "__main__":
    eta, lmbda = optimize_eta()
    print("Optimal eta:", eta)
    print("Optimal lambda:", lmbda)