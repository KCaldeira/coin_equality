import math
from scipy.optimize import root_scalar
from constants import EPSILON

# --- basic maps ---

def a_from_G(G):  # Pareto index a from Gini
    if not (0 < G < 1):
        raise ValueError("G must be in (0,1).")
    return (1.0 + 1.0/G) / 2.0

def G_from_a(a):  # Gini from Pareto index a (inverse of a_from_G)
    if a <= 1:
        raise ValueError("a must be > 1 for finite Gini.")
    return 1.0 / (2.0 * a - 1.0)

def L_pareto(F, G):  # Lorenz curve at F for Pareto-Lorenz with G
    a = a_from_G(G)
    return 1.0 - (1.0 - F)**(1.0 - 1.0/a)

def L_pareto_derivative(F, G):  # Derivative of Lorenz curve dL/dF at F for Pareto-Lorenz with G
    a = a_from_G(G)
    return (1.0 - 1.0/a) * (1.0 - F)**(-1.0/a)

def crossing_rank_from_G(Gini_initial, G2):
    if Gini_initial == G2:
        return 0.5
    r = ((1.0 - G2) * (1.0 + Gini_initial)) / ((1.0 + G2) * (1.0 - Gini_initial))
    s = ((1.0 + Gini_initial) * (1.0 + G2)) / (2.0 * (G2 - Gini_initial))
    return 1.0 - (r ** s)

def _phi(r):  # helper for bracketing cap; φ(r) = (r-1) r^{1/(r-1)-1}
    if r <= 0:
        return float("-inf")
    if abs(r - 1.0) < EPSILON:
        return 0.0
    sgn = 1.0 if r > 1.0 else -1.0
    log_abs = math.log(abs(r - 1.0)) + (1.0/(r - 1.0) - 1.0) * math.log(r)
    return sgn * math.exp(log_abs)

def G2_from_deltaL(deltaL, Gini_initial):
    """
    Solve ΔL(Gini_initial,G2)=deltaL for G2 (bounded in (0,Gini_initial]).
    Caps at G2=0 if deltaL exceeds the Pareto-family maximum.
    """
    if not (0 < Gini_initial < 1):
        raise ValueError("Gini_initial must be in (0,1).")

    if abs(deltaL) < EPSILON:
        return Gini_initial, 0.0

    A1 = (1.0 - Gini_initial) / (1.0 + Gini_initial)
    r_max = 1.0 / A1  # corresponds to G2 -> 0
    deltaL_max = _phi(r_max)
    if deltaL >= deltaL_max - EPSILON:
        return 0.0, float(deltaL - deltaL_max)  # cap & remainder
    # bracket r
    bracket = (1.0 + EPSILON, r_max) if deltaL > 0 else (EPSILON, 1.0 - EPSILON)
    sol = root_scalar(lambda r: _phi(r) - deltaL, bracket=bracket, method="brentq")
    if not sol.converged:
        raise RuntimeError("root_scalar failed for r.")
    r = sol.root
    A2 = r * A1
    A2 = min(max(A2, EPSILON), 1.0)
    G2 = (1.0 - A2) / (1.0 + A2)
    return float(G2), 0.0

# --- the two-step “Pareto-preserving” effective Gini ---

def calculate_Gini_effective_redistribute_abate(f, deltaL, Gini_initial):
    """
    Step 1: find G2 (full redistribution) from ΔL and Gini_initial.
    Step 2: keep the same crossing F*, compute ΔL_eff for partial allocation,
            then solve for G2_eff from ΔL_eff in the Pareto family.
    Returns (G2_eff, remainder_from_cap).
    """
    if not (0 <= f <= 1):
        raise ValueError("f must be in [0,1].")
    # Step 1: full redistribution target in Pareto family
    G2_full, rem = G2_from_deltaL(deltaL, Gini_initial)
    if rem > 0:
        # You already hit the G2=0 cap with full ΔL; partial will remain at/above 0.
        return 0.0, rem

    # Crossing rank for (Gini_initial -> G2_full)
    Fstar = crossing_rank_from_G(Gini_initial, G2_full)
    L1_star = L_pareto(Fstar, Gini_initial)

    # Step 2: partial allocation: ΔL_eff at the same F*
    # L_new(F*) = [ L1_star + (1-f)ΔL ] / (1 - fΔL)
    # ΔL_eff = L_new(F*) - L1_star
    deltaL_eff = deltaL * ((1.0 - f) + f * L1_star) / (1.0 - f * deltaL)

    # Solve for Pareto-equivalent G2_eff
    G2_eff, rem_eff = G2_from_deltaL(deltaL_eff, Gini_initial)
    return G2_eff, rem_eff
