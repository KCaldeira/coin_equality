import math
from scipy.optimize import root_scalar

# --- basic maps ---

def a_from_G(G):  # Pareto index a from Gini
    if not (0 < G < 1):
        raise ValueError("G must be in (0,1).")
    return (1.0 + 1.0/G) / 2.0

def L_pareto(F, G):  # Lorenz curve at F for Pareto-Lorenz with G
    a = a_from_G(G)
    return 1.0 - (1.0 - F)**(1.0 - 1.0/a)

def crossing_rank_from_G(G1, G2):
    if G1 == G2:
        return 0.5
    r = ((1.0 - G2) * (1.0 + G1)) / ((1.0 + G2) * (1.0 - G1))
    s = ((1.0 + G1) * (1.0 + G2)) / (2.0 * (G2 - G1))
    return 1.0 - (r ** s)

def deltaL_G1_G2(G1, G2):  # ΔL at the crossing, Pareto→Pareto, log-stable
    N = (1.0 - G2) * (1.0 + G1)
    D = (1.0 + G2) * (1.0 - G1)
    num = N - D
    r = N / D
    # sign * exp( log|num| - log D + (N/(D-N)) * log r )
    sign = 1.0 if num >= 0 else -1.0
    return sign * math.exp(math.log(abs(num)) - math.log(D) + (N / (D - N)) * math.log(r))

def _phi(r):  # helper for bracketing cap; φ(r) = (r-1) r^{1/(r-1)-1}
    if r <= 0:
        return float("-inf")
    if abs(r - 1.0) < 1e-16:
        return 0.0
    sgn = 1.0 if r > 1.0 else -1.0
    log_abs = math.log(abs(r - 1.0)) + (1.0/(r - 1.0) - 1.0) * math.log(r)
    return sgn * math.exp(log_abs)

def G2_from_deltaL(deltaL, G1):
    """
    Solve ΔL(G1,G2)=deltaL for G2 (bounded in (0,G1]).
    Caps at G2=0 if deltaL exceeds the Pareto-family maximum.
    """
    if not (0 < G1 < 1):
        raise ValueError("G1 must be in (0,1).")
    A1 = (1.0 - G1) / (1.0 + G1)
    r_max = 1.0 / A1  # corresponds to G2 -> 0
    deltaL_max = _phi(r_max)
    if deltaL >= deltaL_max - 1e-15:
        return 0.0, float(deltaL - deltaL_max)  # cap & remainder
    # bracket r
    bracket = (1.0 + 1e-12, r_max) if deltaL > 0 else (1e-12, 1.0 - 1e-12)
    sol = root_scalar(lambda r: _phi(r) - deltaL, bracket=bracket, method="brentq")
    if not sol.converged:
        raise RuntimeError("root_scalar failed for r.")
    r = sol.root
    A2 = r * A1
    A2 = min(max(A2, 1e-15), 1.0)
    G2 = (1.0 - A2) / (1.0 + A2)
    return float(G2), 0.0

# --- the two-step “Pareto-preserving” effective Gini ---

def G2_effective_pareto(f, deltaL, G1):
    """
    Step 1: find G2 (full redistribution) from ΔL and G1.
    Step 2: keep the same crossing F*, compute ΔL_eff for partial allocation,
            then solve for G2_eff from ΔL_eff in the Pareto family.
    Returns (G2_eff, remainder_from_cap).
    """
    if not (0 <= f <= 1):
        raise ValueError("f must be in [0,1].")
    # Step 1: full redistribution target in Pareto family
    G2_full, rem = G2_from_deltaL(deltaL, G1)
    if rem > 0:
        # You already hit the G2=0 cap with full ΔL; partial will remain at/above 0.
        return 0.0, rem

    # Crossing rank for (G1 -> G2_full)
    Fstar = crossing_rank_from_G(G1, G2_full)
    L1_star = L_pareto(Fstar, G1)

    # Step 2: partial allocation: ΔL_eff at the same F*
    # L_new(F*) = [ L1_star + (1-f)ΔL ] / (1 - fΔL)
    # ΔL_eff = L_new(F*) - L1_star
    deltaL_eff = deltaL * ((1.0 - f) + f * L1_star) / (1.0 - f * deltaL)

    # Solve for Pareto-equivalent G2_eff
    G2_eff, rem_eff = G2_from_deltaL(deltaL_eff, G1)
    return G2_eff, rem_eff
