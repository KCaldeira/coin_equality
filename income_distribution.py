import math
import numpy as np
from scipy.optimize import root_scalar
from constants import EPSILON, LOOSE_EPSILON

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
        raise ValueError("Gini_initial must be in (0,1). Invalid value: {}".format(Gini_initial))

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

def calculate_Gini_effective_redistribute_abate(f, deltaL, Gini_climate):
    """
    Step 1: find G2 (full redistribution) from ΔL and Gini_climate.
    Step 2: keep the same crossing F*, compute ΔL_eff for partial allocation,
            then solve for G2_eff from ΔL_eff in the Pareto family.
    Returns (G2_eff, remainder_from_cap).
    """
    if not (0 <= f <= 1):
        raise ValueError("f must be in [0,1].")
    # Step 1: full redistribution target in Pareto family
    G2_full, rem = G2_from_deltaL(deltaL, Gini_climate)
    if rem > 0:
        # You already hit the G2=0 cap with full ΔL; partial will remain at/above 0.
        return 0.0, rem

    # Crossing rank for (Gini_climate -> G2_full)
    Fstar = crossing_rank_from_G(Gini_climate, G2_full)
    L1_star = L_pareto(Fstar, Gini_climate)

    # Step 2: partial allocation: ΔL_eff at the same F*
    # L_new(F*) = [ L1_star + (1-f)ΔL ] / (1 - fΔL)
    # ΔL_eff = L_new(F*) - L1_star
    deltaL_eff = deltaL * ((1.0 - f) + f * L1_star) / (1.0 - f * deltaL)

    # Solve for Pareto-equivalent G2_eff
    G2_eff, rem_eff = G2_from_deltaL(deltaL_eff, Gini_climate)
    return G2_eff, rem_eff


# --- Income at rank F after damage ---

def y_of_F_after_damage(F, Fmin, Fmax, y_mean_before_damage, omega_base, y_damage_distribution_scale, uniform_redistribution, gini, branch=0):
    """
    Compute c(F) from the implicit equation

        c(F) = (1-s) * y_mean_before_damage * dL/dF(F; gini) + uniform_redistribution - omega_base * exp(-c(F) / y_damage_distribution_scale),

    where the Lorenz curve is Pareto with Gini index gini:

        L(F) = 1 - (1-F)^(1 - 1/a),
        a    = (1 + 1/gini)/2,
        dL/dF(F) = (1 - 1/a) * (1 - F)^(-1/a).

    The closed-form solution is:

        A(F) = y_mean_before_damage * dL/dF(F; gini) + uniform_redistribution

        c(F) = A(F) + y_damage_distribution_scale * W(
                     - (omega_base / y_damage_distribution_scale) * exp(-A(F) / y_damage_distribution_scale)
               )

    where W is the Lambert W function (principal branch by default).

    Parameters
    ----------
    F : float or array-like
        Population rank(s) in [0,1].
    Fmin : float
        Minimum population rank for income in [0,1].
    Fmax : float
        Maximum population rank for income in [0,1].
    y_mean_before_damage : float
        Mean income.
    omega_base : float
        Maximum damage scale.
    y_damage_distribution_scale : float
        Damage scale parameter in exp(-c/y_damage_distribution_scale).
    uniform_redistribution : float
        Additive constant in A(F).
    gini : float
        Gini index (0 < gini < 1).
    branch : int, optional
        Lambert W branch index (default 0 = principal).

    Returns
    -------
    y_of_F : float or ndarray
        y_of_F(F) evaluated at the given F values (real part).
    """
    import numpy as np
    from scipy.special import lambertw

    F = np.clip(np.asarray(F), Fmin, Fmax)

    # Pareto-Lorenz shape parameter from Gini
    a = (1.0 + 1.0 / gini) / 2.0

    # dL/dF(F) for Pareto-Lorenz
    dLdF = (1.0 - 1.0 / a) * (1.0 - F) ** (-1.0 / a)

    # A(F)
    A =  y_mean_before_damage * dLdF + uniform_redistribution

    # Argument to Lambert W
    z = - (omega_base / y_damage_distribution_scale) * np.exp(-A / y_damage_distribution_scale)

    # Lambert W (may be complex in general; we usually take the real part)
    W_vals = lambertw(z, k=branch)

    # c(F)
    y_of_F_vals = A + y_damage_distribution_scale * W_vals

    return np.real(y_of_F_vals)


def segment_integral_with_cut(
    Flo,
    Fhi,
    Fcut,
    Fmin,
    Fmax_for_clip,
    y_mean_before_damage,
    omega_base,
    y_damage_distribution_scale,
    uniform_redistribution,
    gini,
    xi,
    wi,
    branch=0,
    cut_at="upper",
):
    """
    Compute ∫_{Flo}^{Fhi} [ y(F; Fmin, Fmax_for_clip, ...) - y(Fcut; Fmin, Fmax_for_clip, ...) ] dF

    Generic function for computing integrals over income distribution segments with a reference cut.
    Used for both taxation (upper tail) and redistribution (lower tail) calculations.

    Parameters
    ----------
    Flo : float
        Lower integration bound.
    Fhi : float
        Upper integration bound.
    Fcut : float
        Rank where reference income y(Fcut) is evaluated.
    Fmin : float
        Minimum rank for clipping in y_of_F_after_damage.
    Fmax_for_clip : float
        Maximum rank for clipping in y_of_F_after_damage.
    y_mean_before_damage : float
        Mean income before damage.
    omega_base : float
        Base climate damage parameter.
    y_damage_distribution_scale : float
        Damage distribution scale parameter.
    uniform_redistribution : float
        Uniform per-capita redistribution amount.
    gini : float
        Gini coefficient.
    xi : ndarray
        Gauss-Legendre quadrature nodes on [-1, 1].
    wi : ndarray
        Gauss-Legendre quadrature weights.
    branch : int, optional
        Lambert W branch (default 0).
    cut_at : str, optional
        Semantic label: "upper" for taxation, "lower" for redistribution (default "upper").

    Returns
    -------
    float
        Integral value.
    """
    # Map Gauss-Legendre nodes from [-1, 1] to [Flo, Fhi]
    F_nodes = 0.5 * (Fhi - Flo) * xi + 0.5 * (Fhi + Flo)
    w_nodes = 0.5 * (Fhi - Flo) * wi

    # y(F) over the segment, using same Fmin and Fmax_for_clip
    y_vals = y_of_F_after_damage(
        F_nodes,
        Fmin,
        Fmax_for_clip,
        y_mean_before_damage,
        omega_base,
        y_damage_distribution_scale,
        uniform_redistribution,
        gini,
        branch=branch,
    )

    # reference value y(Fcut)
    y_cut = y_of_F_after_damage(
        Fcut,
        Fmin,
        Fmax_for_clip,
        y_mean_before_damage,
        omega_base,
        y_damage_distribution_scale,
        uniform_redistribution,
        gini,
        branch=branch,
    )

    integrand = y_vals - y_cut
    integral_val = np.dot(w_nodes, integrand)

    return integral_val


def total_tax_top(
    Fmax,
    Fmin,
    y_mean_before_damage,
    omega_base,
    y_damage_distribution_scale,
    uniform_redistribution,
    gini,
    xi,
    wi,
    target_tax=0.0,
    branch=0,
):
    """
    Compute ∫_{Fmax}^{1} [ y(F; Fmin, 1, ...) - y(Fmax; Fmin, Fmax, ...) ] dF - target_tax

    This is the function we will set to zero in root finding for taxation.

    Parameters
    ----------
    Fmax : float
        Upper boundary for taxation (income ranks above Fmax are taxed).
    Fmin : float
        Lower boundary for income distribution.
    y_mean_before_damage : float
        Mean income before damage.
    omega_base : float
        Base climate damage parameter.
    y_damage_distribution_scale : float
        Damage distribution scale parameter.
    uniform_redistribution : float
        Uniform per-capita redistribution amount.
    gini : float
        Gini coefficient.
    xi : ndarray
        Gauss-Legendre quadrature nodes on [-1, 1].
    wi : ndarray
        Gauss-Legendre quadrature weights.
    target_tax : float, optional
        Target tax amount to subtract (default 0.0).
    branch : int, optional
        Lambert W branch (default 0).

    Returns
    -------
    float
        Integral value minus target_tax (for root finding).
    """
    integral_val = segment_integral_with_cut(
        Flo=Fmax,
        Fhi=1.0,
        Fcut=Fmax,
        Fmin=Fmin,
        Fmax_for_clip=1.0,  # we want F clipped to [Fmin, 1]
        y_mean_before_damage=y_mean_before_damage,
        omega_base=omega_base,
        y_damage_distribution_scale=y_damage_distribution_scale,
        uniform_redistribution=uniform_redistribution,
        gini=gini,
        xi=xi,
        wi=wi,
        branch=branch,
        cut_at="upper",
    )

    return integral_val - target_tax


def total_tax_bottom(
    Fmin,
    y_mean_before_damage,
    omega_base,
    y_damage_distribution_scale,
    uniform_redistribution,
    gini,
    xi,
    wi,
    target_subsidy=0.0,
    branch=0,
):
    """
    Compute ∫_{0}^{Fmin} [ y(Fmin; 0, Fmin, ...) - y(F; 0, Fmin, ...) ] dF - target_subsidy

    This is the function we will set to zero in root finding for redistribution.

    Parameters
    ----------
    Fmin : float
        Lower boundary for redistribution (income ranks below Fmin receive redistribution).
    y_mean_before_damage : float
        Mean income before damage.
    omega_base : float
        Base climate damage parameter.
    y_damage_distribution_scale : float
        Damage distribution scale parameter.
    uniform_redistribution : float
        Uniform per-capita redistribution amount.
    gini : float
        Gini coefficient.
    xi : ndarray
        Gauss-Legendre quadrature nodes on [-1, 1].
    wi : ndarray
        Gauss-Legendre quadrature weights.
    target_subsidy : float, optional
        Target subsidy amount to subtract (default 0.0).
    branch : int, optional
        Lambert W branch (default 0).

    Returns
    -------
    float
        Integral value minus target_subsidy (for root finding).
    """
    integral_val = segment_integral_with_cut(
        Flo=0.0,
        Fhi=Fmin,
        Fcut=Fmin,
        Fmin=0.0,              # model Fmin as bottom of support
        Fmax_for_clip=Fmin,    # clip inside [0, Fmin]
        y_mean_before_damage=y_mean_before_damage,
        omega_base=omega_base,
        y_damage_distribution_scale=y_damage_distribution_scale,
        uniform_redistribution=uniform_redistribution,
        gini=gini,
        xi=xi,
        wi=wi,
        branch=branch,
        cut_at="lower",
    )

    return integral_val - target_subsidy


def find_Fmax(Fmin,
              y_mean_before_damage,
              omega_base,
              y_damage_distribution_scale,
              uniform_redistribution,
              gini,
              xi,
              wi,
              target_tax=0.0,
              branch=0,
              tol=LOOSE_EPSILON):
    """
    Find Fmax in [Fmin, 1) such that total_tax_top(Fmax) = target_tax.

    Uses a bracketing root-finder.

    Parameters
    ----------
    Fmin : float
        Lower boundary for income distribution.
    y_mean_before_damage : float
        Mean income before damage.
    omega_base : float
        Base climate damage parameter.
    y_damage_distribution_scale : float
        Damage distribution scale parameter.
    uniform_redistribution : float
        Uniform per-capita redistribution amount.
    gini : float
        Gini coefficient.
    xi : ndarray
        Gauss-Legendre quadrature nodes on [-1, 1].
    wi : ndarray
        Gauss-Legendre quadrature weights.
    target_tax : float, optional
        Target tax amount (default 0.0).
    branch : int, optional
        Lambert W branch (default 0).
    tol : float, optional
        Tolerance for root finding (default LOOSE_EPSILON = 1e-8).

    Returns
    -------
    float
        Fmax value such that total_tax_top(Fmax) = target_tax.
    """
    # Define a wrapper with all parameters bound
    def f(Fmax):
        return total_tax_top(
            Fmax,
            Fmin,
            y_mean_before_damage,
            omega_base,
            y_damage_distribution_scale,
            uniform_redistribution,
            gini,
            xi,
            wi,
            target_tax=target_tax,
            branch=branch,
        )

    # Bracket Fmax between Fmin and something close to 1
    left = Fmin
    right = 0.999999

    f_left = f(left)
    f_right = f(right)

    if f_left * f_right > 0:
        raise RuntimeError(
            f"Root not bracketed: total_tax_top(Fmin)={f_left}, total_tax_top(0.999999)={f_right}"
        )

    sol = root_scalar(f, bracket=[left, right], method="brentq", xtol=tol)
    if not sol.converged:
        raise RuntimeError("root_scalar did not converge for find_Fmax")

    return sol.root


def find_Fmin(y_mean_before_damage,
              omega_base,
              y_damage_distribution_scale,
              uniform_redistribution,
              gini,
              xi,
              wi,
              target_subsidy=0.0,
              branch=0,
              tol=LOOSE_EPSILON):
    """
    Find Fmin in (0, 1) such that total_tax_bottom(Fmin) = target_subsidy.

    Uses a bracketing root-finder.

    Parameters
    ----------
    y_mean_before_damage : float
        Mean income before damage.
    omega_base : float
        Base climate damage parameter.
    y_damage_distribution_scale : float
        Damage distribution scale parameter.
    uniform_redistribution : float
        Uniform per-capita redistribution amount.
    gini : float
        Gini coefficient.
    xi : ndarray
        Gauss-Legendre quadrature nodes on [-1, 1].
    wi : ndarray
        Gauss-Legendre quadrature weights.
    target_subsidy : float, optional
        Target subsidy amount (default 0.0).
    branch : int, optional
        Lambert W branch (default 0).
    tol : float, optional
        Tolerance for root finding (default LOOSE_EPSILON = 1e-8).

    Returns
    -------
    float
        Fmin value such that total_tax_bottom(Fmin) = target_subsidy.
    """
    # Define a wrapper with all parameters bound
    def f(Fmin):
        return total_tax_bottom(
            Fmin,
            y_mean_before_damage,
            omega_base,
            y_damage_distribution_scale,
            uniform_redistribution,
            gini,
            xi,
            wi,
            target_subsidy=target_subsidy,
            branch=branch,
        )

    # Bracket Fmin between something close to 0 and something less than 1
    left = 0.000001
    right = 0.999999

    f_left = f(left)
    f_right = f(right)

    if f_left * f_right > 0:
        raise RuntimeError(
            f"Root not bracketed: total_tax_bottom(0.000001)={f_left}, total_tax_bottom(0.999999)={f_right}"
        )

    sol = root_scalar(f, bracket=[left, right], method="brentq", xtol=tol)
    if not sol.converged:
        raise RuntimeError("root_scalar did not converge for find_Fmin")

    return sol.root
