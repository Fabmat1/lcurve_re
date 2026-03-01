// physical_prior.h
// ═══════════════════════════════════════════════════════════════════════
//  Analytical physical prior for binary-star LCURVE MCMC fitting.
//
//  Replaces the grid-based mass_ratio_pdf approach. Given proposed
//  LCURVE model parameters (i, q, velocity_scale, r1), derives the
//  implied physical observables (K₁, R₁, M₁, M₂) and evaluates
//  Gaussian priors against measured values.
//
//  All correlations are captured exactly because every implied
//  observable is computed from the full joint parameter vector.
// ═══════════════════════════════════════════════════════════════════════
#pragma once

#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>

// ─── Observed constraints from spectroscopy / SED analysis ───────────
struct ObservedConstraints {
    // RV semi-amplitude of primary K₁  [km/s]
    double K_obs = 0, K_err_lo = 0, K_err_hi = 0;
    bool   has_K = false;

    // Primary stellar mass M₁  [M_sun]
    double M1_obs = 0, M1_err_lo = 0, M1_err_hi = 0;
    bool   has_M1 = false;

    // Minimum companion mass M₂,min from mass function  [M_sun]
    // Only used as a soft one-sided constraint (M₂ ≥ M₂,min).
    double M2min_obs = 0, M2min_err_lo = 0, M2min_err_hi = 0;
    bool   has_M2min = false;

    // Physical radius of primary R₁  [R_sun]
    // (NB: config key may be called "r1", but this is the physical
    //  radius, NOT the LCURVE fractional-radius model parameter.)
    double R1_obs = 0, R1_err_lo = 0, R1_err_hi = 0;
    bool   has_R1 = false;

    // Orbital period [days] — treated as exact
    double P_days = 1.0;

    // Geometric sin(i) prior for randomly oriented orbits
    bool use_sin_i_prior = true;
};

namespace PhysicalPrior {

// ─── Physical constants ──────────────────────────────────────────────
inline constexpr double DEG2RAD = M_PI / 180.0;
inline constexpr double DAY2SEC = 86400.0;
inline constexpr double RSUN_KM = 695700.0;
// G × M_sun  [km³ s⁻²]
//   G = 6.67430e-11 m³/(kg s²) = 6.67430e-20 km³/(kg s²)
//   M_sun = 1.98892e30 kg
//   → G M_sun = 1.32713e11 km³ s⁻²
inline constexpr double G_MSUN  = 1.32713e11;

// ─── Log-PDF helpers ─────────────────────────────────────────────────
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

// ─── Main prior evaluation ───────────────────────────────────────────
//
//  LCURVE model parameters:
//    i_deg  – orbital inclination  [degrees]
//    q      – mass ratio  M₂ / M₁
//    vs     – velocity_scale = 2πa/P  (relative orbital speed)  [km/s]
//    r1     – fractional primary radius  R₁ / a
//
//  Derived observables:
//    K₁        = vs · sin i · q / (1+q)                  [km/s]
//    R₁        = r₁ · a  =  r₁ · vs · P / (2π)          [km → R☉]
//    M_total   = 4π² a³ / (G P²)                         [M☉]
//    M₁        = M_total / (1+q)
//    M₂        = q · M₁
//
inline double compute(double i_deg, double q, double vs, double r1,
                      const ObservedConstraints& c)
{
    const double sin_i = std::sin(i_deg * DEG2RAD);

    // Reject manifestly unphysical proposals immediately
    if (sin_i < 1e-10 || q <= 0.0 || vs <= 0.0
        || r1 <= 0.0   || r1 >= 1.0)
        return -1e30;

    const double P_s  = c.P_days * DAY2SEC;
    const double a_km = vs * P_s / (2.0 * M_PI);
    double lp = 0.0;

    // ── Geometric inclination prior:  p(i) ∝ sin i ──────────────────
    if (c.use_sin_i_prior)
        lp += std::log(sin_i);

    // ── 1. K₁ constraint ─────────────────────────────────────────────
    if (c.has_K) {
        double K_impl = vs * sin_i * q / (1.0 + q);
        lp += log_split_gauss(K_impl, c.K_obs, c.K_err_lo, c.K_err_hi);
    }

    // ── 2. R₁ constraint ─────────────────────────────────────────────
    if (c.has_R1) {
        double R1_impl = r1 * a_km / RSUN_KM;          // R☉
        lp += log_split_gauss(R1_impl, c.R1_obs, c.R1_err_lo, c.R1_err_hi);
    }

    // ── 3. M₁ constraint (from Kepler's 3rd law) ────────────────────
    const double M_total =
        4.0 * M_PI * M_PI * a_km * a_km * a_km / (G_MSUN * P_s * P_s);
    const double M1_impl = M_total / (1.0 + q);
    const double M2_impl = q * M1_impl;

    if (c.has_M1)
        lp += log_split_gauss(M1_impl, c.M1_obs, c.M1_err_lo, c.M1_err_hi);

    // ── 4. M₂ ≥ M₂,min  soft one-sided constraint ──────────────────
    //  M₂,min is the companion mass at i = 90°; the true M₂ must
    //  exceed it.  Apply a half-Gaussian penalty for violations.
    if (c.has_M2min && M2_impl < c.M2min_obs) {
        double sig = (c.M2min_err_lo > 0.0) ? c.M2min_err_lo : c.M2min_err_hi;
        if (sig <= 0.0) sig = 0.01 * c.M2min_obs;      // 1 % fallback
        double z = (M2_impl - c.M2min_obs) / sig;
        lp += -0.5 * z * z;
    }

    return lp;
}

// ─── Diagnostic printer ──────────────────────────────────────────────
inline void print_implied(double i_deg, double q, double vs, double r1,
                           const ObservedConstraints& c)
{
    const double sin_i = std::sin(i_deg * DEG2RAD);
    const double P_s   = c.P_days * DAY2SEC;
    const double a_km  = vs * P_s / (2.0 * M_PI);

    double K_impl  = vs * sin_i * q / (1.0 + q);
    double R1_impl = r1 * a_km / RSUN_KM;
    double M_total = 4.0*M_PI*M_PI * a_km*a_km*a_km / (G_MSUN * P_s*P_s);
    double M1_impl = M_total / (1.0 + q);
    double M2_impl = q * M1_impl;

    auto fl = std::cout.flags();
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "  K1  (implied) = " << K_impl  << " km/s";
    if (c.has_K) std::cout << "   (obs " << c.K_obs
              << " -" << c.K_err_lo << "/+" << c.K_err_hi << ")";
    std::cout << "\n";

    std::cout << "  R1  (implied) = " << R1_impl << " R_sun";
    if (c.has_R1) std::cout << "   (obs " << c.R1_obs
              << " -" << c.R1_err_lo << "/+" << c.R1_err_hi << ")";
    std::cout << "\n";

    std::cout << "  M1  (implied) = " << M1_impl << " M_sun";
    if (c.has_M1) std::cout << "   (obs " << c.M1_obs
              << " -" << c.M1_err_lo << "/+" << c.M1_err_hi << ")";
    std::cout << "\n";

    std::cout << "  M2  (implied) = " << M2_impl << " M_sun";
    if (c.has_M2min) std::cout << "   (M2_min " << c.M2min_obs << ")";
    std::cout << "\n";

    std::cout << "  M_total       = " << M_total << " M_sun\n";
    std::cout << "  a             = " << a_km    << " km  =  "
              << (a_km / RSUN_KM) << " R_sun\n";

    std::cout.flags(fl);
}

} // namespace PhysicalPrior