// physical_prior.h
#pragma once

#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <tuple>
#include <stdexcept>

struct ObservedConstraints {
    // ── Radial velocity semi-amplitudes ──
    double K_obs = 0, K_err_lo = 0, K_err_hi = 0;
    bool   has_K = false;

    double K2_obs = 0, K2_err_lo = 0, K2_err_hi = 0;
    bool   has_K2 = false;

    // ── Masses ──
    double M1_obs = 0, M1_err_lo = 0, M1_err_hi = 0;
    bool   has_M1 = false;

    double M2_obs = 0, M2_err_lo = 0, M2_err_hi = 0;
    bool   has_M2 = false;

    double M2min_obs = 0, M2min_err_lo = 0, M2min_err_hi = 0;
    bool   has_M2min = false;

    double Mtotal_obs = 0, Mtotal_err_lo = 0, Mtotal_err_hi = 0;
    bool   has_Mtotal = false;

    // ── Mass ratio ──
    double q_obs = 0, q_err_lo = 0, q_err_hi = 0;
    bool   has_q = false;

    // ── Radii ──
    double R1_obs = 0, R1_err_lo = 0, R1_err_hi = 0;
    bool   has_R1 = false;

    double R2_obs = 0, R2_err_lo = 0, R2_err_hi = 0;
    bool   has_R2 = false;

    // ── Surface gravities ──
    double logg1_obs = 0, logg1_err_lo = 0, logg1_err_hi = 0;
    bool   has_logg1 = false;

    double logg2_obs = 0, logg2_err_lo = 0, logg2_err_hi = 0;
    bool   has_logg2 = false;

    // ── Effective temperatures ──
    double T1_obs = 0, T1_err_lo = 0, T1_err_hi = 0;
    bool   has_T1 = false;

    double T2_obs = 0, T2_err_lo = 0, T2_err_hi = 0;
    bool   has_T2 = false;

    double P_days = 1.0;
    bool use_sin_i_prior = true;

    // ── Prior weighting ──
    // Multiplier on the log-prior.  At weight=1 (default), one prior
    // σ is worth exactly one data-point σ.  Increase to make priors
    // harder to violate.  A value of N_data/N_priors is a reasonable
    // "unit information" scaling.
    double prior_weight = 1.0;
};

namespace PhysicalPrior {

inline constexpr double DEG2RAD = M_PI / 180.0;
inline constexpr double DAY2SEC = 86400.0;
inline constexpr double RSUN_KM = 695700.0;

// IAU 2015 nominal solar mass parameter  (exact by convention)
//   GM_sun^N = 1.327 124 4 × 10²⁰  m³ s⁻²
//            = 1.327 124 4 × 10¹¹  km³ s⁻²
inline constexpr double G_MSUN  = 1.3271244e11;

// Solar surface gravity:  log10( G·M_sun / R_sun² )  in cgs
//   G_cgs   = 6.67430e-8  cm³ g⁻¹ s⁻²
//   M_sun_g = 1.98892e33  g
//   R_sun_cm= 6.9570e10   cm
//   → GM/R² = 2.7420e4 cm s⁻²  → log10 = 4.4380
inline constexpr double LOGG_SUN = 4.4380;

inline double log_gauss(double x, double mu, double sigma)
{
    if (sigma <= 0.0) return 0.0;
    double z = (x - mu) / sigma;
    return -0.5 * z * z - 0.5 * std::log(2.0 * M_PI) - std::log(sigma);
}

inline double log_split_gauss(double x, double mu,
                               double sig_lo, double sig_hi)
{
    if (sig_lo <= 0.0 && sig_hi <= 0.0) return 0.0;
    double sigma = (x < mu) ? sig_lo : sig_hi;
    if (sigma <= 0.0) sigma = std::max(sig_lo, sig_hi);
    return log_gauss(x, mu, sigma);
}

// ═════════════════════════════════════════════════════════════════════
//  Main prior computation.
//
//  Parameters passed in from the current MCMC state:
//    i_deg  – orbital inclination  [degrees]
//    q      – mass ratio  M2/M1
//    vs     – velocity_scale  [km/s]
//    r1     – fractional radius of star 1  (R1/a)
//    r2     – fractional radius of star 2  (R2/a)
//    t1     – effective temperature of star 1  [K]
//    t2     – effective temperature of star 2  [K]
//
//  Only constraints whose  has_*  flag is true contribute.
// ═════════════════════════════════════════════════════════════════════
inline double compute(double i_deg, double q, double vs, double r1,
                      double r2, double t1, double t2,
                      const ObservedConstraints& c)
{
    const double sin_i = std::sin(i_deg * DEG2RAD);
    if (sin_i < 1e-10 || q <= 0.0 || vs <= 0.0
        || r1 <= 0.0   || r1 >= 1.0)
        return -1e30;

    const double P_s  = c.P_days * DAY2SEC;
    const double a_km = vs * P_s / (2.0 * M_PI);
    double lp_geometric = 0.0;
    double lp_observational = 0.0;


    // ── K1 = vs·sin(i)·q/(1+q) ──
    if (c.has_K) {
        double K_impl = vs * sin_i * q / (1.0 + q);
        lp_observational += log_split_gauss(K_impl, c.K_obs, c.K_err_lo, c.K_err_hi);
    }

    // ── K2 = vs·sin(i)/(1+q) ──
    if (c.has_K2) {
        double K2_impl = vs * sin_i / (1.0 + q);
        lp_observational += log_split_gauss(K2_impl, c.K2_obs, c.K2_err_lo, c.K2_err_hi);
    }

    // ── R1 = r1·a  [R_sun] ──
    const double R1_impl = r1 * a_km / RSUN_KM;
    if (c.has_R1)
        lp_observational += log_split_gauss(R1_impl, c.R1_obs, c.R1_err_lo, c.R1_err_hi);

    // ── R2 = r2·a  [R_sun] ──
    const double R2_impl = (r2 > 0.0 && r2 < 1.0)
                         ? r2 * a_km / RSUN_KM : 0.0;
    if (c.has_R2 && R2_impl > 0.0)
        lp_observational += log_split_gauss(R2_impl, c.R2_obs, c.R2_err_lo, c.R2_err_hi);

    // ── Masses from Kepler's third law ──
    const double M_total =
        4.0 * M_PI * M_PI * a_km * a_km * a_km / (G_MSUN * P_s * P_s);
    const double M1_impl = M_total / (1.0 + q);
    const double M2_impl = q * M1_impl;

    if (c.has_M1)
        lp_observational += log_split_gauss(M1_impl, c.M1_obs, c.M1_err_lo, c.M1_err_hi);

    if (c.has_M2)
        lp_observational += log_split_gauss(M2_impl, c.M2_obs, c.M2_err_lo, c.M2_err_hi);

    // ── M2_min: one-sided penalty ──
    if (c.has_M2min && M2_impl < c.M2min_obs) {
        double sig = (c.M2min_err_lo > 0.0) ? c.M2min_err_lo : c.M2min_err_hi;
        if (sig <= 0.0) sig = 0.01 * c.M2min_obs;
        double z = (M2_impl - c.M2min_obs) / sig;
        lp_observational += -0.5 * z * z;
    }

    // ── Total mass ──
    if (c.has_Mtotal)
        lp_observational += log_split_gauss(M_total, c.Mtotal_obs,
                              c.Mtotal_err_lo, c.Mtotal_err_hi);

    // ── Mass ratio ──
    if (c.has_q)
        lp_observational += log_split_gauss(q, c.q_obs, c.q_err_lo, c.q_err_hi);

    // ── Surface gravity of star 1 ──
    if (c.has_logg1 && M1_impl > 0.0 && R1_impl > 0.0) {
        double logg1_impl = LOGG_SUN
                          + std::log10(M1_impl) - 2.0 * std::log10(R1_impl);
        lp_observational += log_split_gauss(logg1_impl, c.logg1_obs,
                              c.logg1_err_lo, c.logg1_err_hi);
    }

    // ── Surface gravity of star 2 ──
    if (c.has_logg2 && M2_impl > 0.0 && R2_impl > 0.0) {
        double logg2_impl = LOGG_SUN
                          + std::log10(M2_impl) - 2.0 * std::log10(R2_impl);
        lp_observational += log_split_gauss(logg2_impl, c.logg2_obs,
                              c.logg2_err_lo, c.logg2_err_hi);
    }

    // ── Effective temperatures ──
    if (c.has_T1 && t1 > 0.0)
        lp_observational += log_split_gauss(t1, c.T1_obs, c.T1_err_lo, c.T1_err_hi);

    if (c.has_T2 && t2 > 0.0)
        lp_observational += log_split_gauss(t2, c.T2_obs, c.T2_err_lo, c.T2_err_hi);

    return lp_geometric + c.prior_weight * lp_observational;
}

// ═════════════════════════════════════════════════════════════════════
//  Decomposed prior: returns (lp_geometric, lp_observational_unweighted)
//  so callers can diagnose the balance between chi² and prior.
// ═════════════════════════════════════════════════════════════════════
inline std::pair<double, double> compute_decomposed(
    double i_deg, double q, double vs, double r1,
    double r2, double t1, double t2,
    const ObservedConstraints& c)
{
    const double sin_i = std::sin(i_deg * DEG2RAD);
    if (sin_i < 1e-10 || q <= 0.0 || vs <= 0.0
        || r1 <= 0.0   || r1 >= 1.0)
        return {-1e30, -1e30};

    // Save weight, compute with weight=1
    ObservedConstraints c_copy = c;
    c_copy.prior_weight = 1.0;

    double lp_total_w1 = compute(i_deg, q, vs, r1, r2, t1, t2, c_copy);
    double lp_geom = 0.0;
    double lp_obs = lp_total_w1 - lp_geom;

    return {lp_geom, lp_obs};
}

// ═════════════════════════════════════════════════════════════════════
//  Solve for (q, vs, r1) exactly consistent with (K₁, M₁, R₁, P)
//  at a given inclination.
//
//  From Kepler + radial velocity:
//    M₁ = K₁³ P (1+q)² / (2π G q³ sin³i)
//
//  Rearranged:
//    (1+q)² / q³ = 2π G M₁ sin³i / (K₁³ P)
//
//  The LHS is monotonically decreasing for q > 0, so there is
//  exactly one root for any positive RHS.
// ═════════════════════════════════════════════════════════════════════
inline bool solve_consistent_params(
    double i_deg,
    double K_target, double M1_target, double R1_target,
    double P_days,
    double& q_out, double& vs_out, double& r1_out)
{
    const double sin_i = std::sin(i_deg * DEG2RAD);
    if (sin_i < 0.01) return false;
    const double P_s   = P_days * DAY2SEC;
    const double sin3i = sin_i * sin_i * sin_i;

    // RHS of (1+q)²/q³ = rhs
    double rhs = 2.0 * M_PI * G_MSUN * M1_target * sin3i
                 / (K_target * K_target * K_target * P_s);

    // Bisection
    double q_lo = 1e-4, q_hi = 1e4;
    for (int iter = 0; iter < 200; ++iter) {
        double q_mid = std::sqrt(q_lo * q_hi);   // geometric mean
        double f = (1.0 + q_mid) * (1.0 + q_mid)
                 / (q_mid * q_mid * q_mid);
        if (f > rhs) q_lo = q_mid;
        else         q_hi = q_mid;
        if (q_hi / q_lo < 1.0 + 1e-12) break;
    }
    q_out = std::sqrt(q_lo * q_hi);

    // velocity_scale from K₁ = vs sin(i) q/(1+q)
    vs_out = K_target * (1.0 + q_out) / (q_out * sin_i);

    // fractional radius from R₁ = r₁ × a,  a = vs P/(2π)
    double a_km = vs_out * P_s / (2.0 * M_PI);
    r1_out = R1_target * RSUN_KM / a_km;

    if (r1_out <= 0.0 || r1_out >= 1.0) return false;

    return true;
}

// ─── Diagnostic: check & print self-consistency ──────────────────────
inline void print_implied(double i_deg, double q, double vs, double r1,
                          double r2, double t1, double t2,
                          const ObservedConstraints& c)
{
    const double sin_i = std::sin(i_deg * DEG2RAD);
    const double P_s   = c.P_days * DAY2SEC;
    const double a_km  = vs * P_s / (2.0 * M_PI);

    double K_impl   = vs * sin_i * q / (1.0 + q);
    double K2_impl  = vs * sin_i / (1.0 + q);
    double R1_impl  = r1 * a_km / RSUN_KM;
    double R2_impl  = (r2 > 0.0 && r2 < 1.0) ? r2 * a_km / RSUN_KM : 0.0;
    double M_total  = 4.0*M_PI*M_PI * a_km*a_km*a_km / (G_MSUN * P_s*P_s);
    double M1_impl  = M_total / (1.0 + q);
    double M2_impl  = q * M1_impl;

    auto fl = std::cout.flags();
    std::cout << std::fixed << std::setprecision(4);

    // ── K1 ──
    std::cout << "  K1  (implied) = " << K_impl  << " km/s";
    if (c.has_K) std::cout << "   (obs " << c.K_obs
              << " -" << c.K_err_lo << "/+" << c.K_err_hi << ")";
    std::cout << "\n";

    // ── K2 ──
    std::cout << "  K2  (implied) = " << K2_impl << " km/s";
    if (c.has_K2) std::cout << "   (obs " << c.K2_obs
              << " -" << c.K2_err_lo << "/+" << c.K2_err_hi << ")";
    std::cout << "\n";

    // ── R1 ──
    std::cout << "  R1  (implied) = " << R1_impl << " R_sun";
    if (c.has_R1) std::cout << "   (obs " << c.R1_obs
              << " -" << c.R1_err_lo << "/+" << c.R1_err_hi << ")";
    std::cout << "\n";

    // ── R2 ──
    if (R2_impl > 0.0) {
        std::cout << "  R2  (implied) = " << R2_impl << " R_sun";
        if (c.has_R2) std::cout << "   (obs " << c.R2_obs
                  << " -" << c.R2_err_lo << "/+" << c.R2_err_hi << ")";
        std::cout << "\n";
    }

    // ── M1 ──
    std::cout << "  M1  (implied) = " << M1_impl << " M_sun";
    if (c.has_M1) std::cout << "   (obs " << c.M1_obs
              << " -" << c.M1_err_lo << "/+" << c.M1_err_hi << ")";
    std::cout << "\n";

    // ── M2 ──
    std::cout << "  M2  (implied) = " << M2_impl << " M_sun";
    if (c.has_M2) std::cout << "   (obs " << c.M2_obs
              << " -" << c.M2_err_lo << "/+" << c.M2_err_hi << ")";
    if (c.has_M2min) std::cout << "   (M2_min " << c.M2min_obs << ")";
    std::cout << "\n";

    // ── q ──
    std::cout << "  q   (model)   = " << q;
    if (c.has_q) std::cout << "   (obs " << c.q_obs
              << " -" << c.q_err_lo << "/+" << c.q_err_hi << ")";
    std::cout << "\n";

    // ── M_total ──
    std::cout << "  M_total       = " << M_total << " M_sun";
    if (c.has_Mtotal) std::cout << "   (obs " << c.Mtotal_obs
              << " -" << c.Mtotal_err_lo << "/+" << c.Mtotal_err_hi << ")";
    std::cout << "\n";

    // ── logg1 ──
    if (M1_impl > 0.0 && R1_impl > 0.0) {
        double logg1 = LOGG_SUN + std::log10(M1_impl)
                     - 2.0 * std::log10(R1_impl);
        std::cout << "  logg1         = " << logg1 << " dex";
        if (c.has_logg1) std::cout << "   (obs " << c.logg1_obs
                  << " -" << c.logg1_err_lo << "/+" << c.logg1_err_hi << ")";
        std::cout << "\n";
    }

    // ── logg2 ──
    if (M2_impl > 0.0 && R2_impl > 0.0) {
        double logg2 = LOGG_SUN + std::log10(M2_impl)
                     - 2.0 * std::log10(R2_impl);
        std::cout << "  logg2         = " << logg2 << " dex";
        if (c.has_logg2) std::cout << "   (obs " << c.logg2_obs
                  << " -" << c.logg2_err_lo << "/+" << c.logg2_err_hi << ")";
        std::cout << "\n";
    }

    // ── T1 ──
    if (t1 > 0.0) {
        std::cout << "  T1  (model)   = " << std::setprecision(0)
                  << t1 << " K";
        if (c.has_T1) std::cout << "   (obs " << c.T1_obs
                  << " -" << c.T1_err_lo << "/+" << c.T1_err_hi << ")";
        std::cout << "\n";
        std::cout << std::setprecision(4);
    }

    // ── T2 ──
    if (t2 > 0.0) {
        std::cout << "  T2  (model)   = " << std::setprecision(0)
                  << t2 << " K";
        if (c.has_T2) std::cout << "   (obs " << c.T2_obs
                  << " -" << c.T2_err_lo << "/+" << c.T2_err_hi << ")";
        std::cout << "\n";
        std::cout << std::setprecision(4);
    }

    // ── Separation ──
    std::cout << "  a             = " << a_km    << " km  =  "
              << (a_km / RSUN_KM) << " R_sun\n";

    // Cross-check: M_total via vs³ (same formula, different path)
    double M_total_v = vs*vs*vs * P_s / (2.0 * M_PI * G_MSUN);
    if (std::abs(M_total - M_total_v) > 1e-6 * M_total)
        std::cout << "  [BUG] M_total inconsistency: "
                  << M_total << " vs " << M_total_v << "\n";

// Flag if K₁ is far from observed (accounting for error bars)
    double K_max = vs * q / (1.0 + q);   // i = 90°
    if (c.has_K) {
        double K_impl = vs * sin_i * q / (1.0 + q);
        double K_sig  = (K_impl < c.K_obs) ? c.K_err_lo : c.K_err_hi;
        if (K_sig <= 0.0) K_sig = std::max(c.K_err_lo, c.K_err_hi);
        if (K_sig <= 0.0) K_sig = 0.1 * c.K_obs;
        double K_pull = std::abs(K_impl - c.K_obs) / K_sig;

        if (K_pull > 3.0) {
            std::cout << "  [WARNING] K1 implied = " << K_impl
                      << " is " << std::fixed << std::setprecision(1)
                      << K_pull << "σ from K_obs = " << c.K_obs
                      << " (−" << c.K_err_lo << "/+" << c.K_err_hi << ")\n";
            if (K_max < c.K_obs - 3.0 * c.K_err_lo) {
                std::cout << "  [WARNING] K1_max(i=90) = " << K_max
                          << " — even edge-on cannot reach >3σ range.\n"
                          << "            Need higher vs or different q.\n";
            }
        }
    }

    // Flag if K₂ is far from observed
    if (c.has_K2) {
        double K2_impl = vs * sin_i / (1.0 + q);
        double K2_max  = vs / (1.0 + q);
        double K2_sig  = (K2_impl < c.K2_obs) ? c.K2_err_lo : c.K2_err_hi;
        if (K2_sig <= 0.0) K2_sig = std::max(c.K2_err_lo, c.K2_err_hi);
        if (K2_sig <= 0.0) K2_sig = 0.1 * c.K2_obs;
        double K2_pull = std::abs(K2_impl - c.K2_obs) / K2_sig;

        if (K2_pull > 3.0) {
            std::cout << "  [WARNING] K2 implied = " << K2_impl
                      << " is " << std::fixed << std::setprecision(1)
                      << K2_pull << "σ from K2_obs = " << c.K2_obs
                      << " (−" << c.K2_err_lo << "/+" << c.K2_err_hi << ")\n";
            if (K2_max < c.K2_obs - 3.0 * c.K2_err_lo) {
                std::cout << "  [WARNING] K2_max(i=90) = " << K2_max
                          << " — even edge-on cannot reach >3σ range.\n";
            }
        }
    }

    std::cout.flags(fl);
}

} // namespace PhysicalPrior