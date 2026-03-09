// levmarq_solver.cpp
// ═══════════════════════════════════════════════════════════════════════
//  Levenberg-Marquardt least-squares solver for LCURVE binary-star
//  light curves, with physical priors treated as augmented residuals.
//
//  The algorithm follows MINPACK's lmder/lmdif strategy:
//    • Column-scaled trust region (Moré 1978)
//    • Gain-ratio adaptive damping (λ adjustment)
//    • Jacobian via forward finite differences, step ~ √ε · |xᵢ|
//    • Convergence on ftol, xtol, gtol (same semantics as MINPACK)
//    • Priors enter as extra rows in the residual vector
//    • Final covariance from (J^T J)^{-1} · s² at the solution
//
//  Key features
//  ────────────
//  • Prior constraints (K1, K2, M1, M2, q, R1, R2, logg, T, sin i)
//    are treated as pseudo-observations ⇒ natural LM incorporation
//  • Asymmetric error bars: uses appropriate σ depending on sign
//  • Bounded parameters with active-set clamping
//  • MINPACK-style diagonal scaling (column norms of J)
//  • Covariance matrix & formal uncertainties at the optimum
//  • Correlation matrix report
//  • Gnuplot live plotting (same interface as mcmc_solver)
// ═══════════════════════════════════════════════════════════════════════

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <cassert>

#include <nlohmann/json.hpp>
#include "../src/lcurve_base/lcurve.h"
#include "../src/new_helpers.h"
#include "../src/new_subs.h"
#include "../src/physical_prior.h"

#include <sys/ioctl.h>
#include <unistd.h>
#include <signal.h>
#include <atomic>

using namespace std;
using json = nlohmann::json;
using Clock = chrono::steady_clock;

// ═══════════════════════ terminal helpers ═════════════════════════════

inline int current_tty_columns()
{
    winsize ws{};
    if (::isatty(STDOUT_FILENO) == 0) return 80;
    if (::ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == -1) return 80;
    return ws.ws_col ? ws.ws_col : 80;
}

std::atomic<int> tty_cols{ current_tty_columns() };
void sigwinch_handler(int) {
    tty_cols.store(current_tty_columns(), std::memory_order_relaxed);
}

// ANSI colours
static const string RESET        = "\033[0m";
static const string BRIGHT_GREEN = "\033[92m";
static const string BRIGHT_BLUE  = "\033[94m";
static const string BRIGHT_YELLOW= "\033[93m";
static const string BRIGHT_RED   = "\033[91m";
static const string BRIGHT_CYAN  = "\033[96m";
static const string BRIGHT_WHITE = "\033[97m";
static const string DIM          = "\033[2m";

// ═══════════════════════ linear algebra helpers ══════════════════════
//
//  Small dense routines — parameter count is O(10), so clarity over
//  BLAS calls.  Everything column-major where it matters.
// ════════════════════════════════════════════════════════════════════

// Solve  (J^T J + λ diag(D²)) δ = -J^T r   via Cholesky
// Returns false if the system is singular (λ too small).
//
//   A  = J^T J   (n×n, overwritten with Cholesky factor)
//   g  = J^T r   (n)
//   D  = diagonal scale (n)
//   λ  = damping
//   δ  = output step (n)
//
static bool solve_normal_equations(vector<vector<double>>& A,
                                   const vector<double>& g,
                                   const vector<double>& D,
                                   double lambda,
                                   vector<double>& delta)
{
    const int n = static_cast<int>(A.size());
    // augment diagonal
    for (int i = 0; i < n; ++i)
        A[i][i] += lambda * D[i] * D[i];

    // Cholesky  A = L L^T
    vector<vector<double>> L(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = A[i][j];
            for (int k = 0; k < j; ++k)
                s -= L[i][k] * L[j][k];
            if (i == j) {
                if (s <= 0.0) return false;
                L[i][j] = std::sqrt(s);
            } else {
                L[i][j] = s / L[j][j];
            }
        }
    }

    // forward solve  L y = -g
    vector<double> y(n);
    for (int i = 0; i < n; ++i) {
        double s = -g[i];
        for (int k = 0; k < i; ++k)
            s -= L[i][k] * y[k];
        y[i] = s / L[i][i];
    }

    // back solve  L^T δ = y
    delta.resize(n);
    for (int i = n - 1; i >= 0; --i) {
        double s = y[i];
        for (int k = i + 1; k < n; ++k)
            s -= L[k][i] * delta[k];
        delta[i] = s / L[i][i];
    }
    return true;
}

// Invert symmetric positive-definite matrix via Cholesky
// Returns false on singularity
static bool invert_spd(const vector<vector<double>>& A,
                       vector<vector<double>>& Ainv)
{
    const int n = static_cast<int>(A.size());
    // Cholesky
    vector<vector<double>> L(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = A[i][j];
            for (int k = 0; k < j; ++k)
                s -= L[i][k] * L[j][k];
            if (i == j) {
                if (s <= 1e-30) return false;
                L[i][j] = std::sqrt(s);
            } else {
                L[i][j] = s / L[j][j];
            }
        }
    }

    // Invert L (lower triangular)
    vector<vector<double>> Linv(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        Linv[i][i] = 1.0 / L[i][i];
        for (int j = i + 1; j < n; ++j) {
            double s = 0.0;
            for (int k = i; k < j; ++k)
                s -= L[j][k] * Linv[k][i];
            Linv[j][i] = s / L[j][j];
        }
    }

    // A^{-1} = L^{-T} L^{-1}
    Ainv.assign(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j) {
            double s = 0.0;
            for (int k = i; k < n; ++k)
                s += Linv[k][i] * Linv[k][j];
            Ainv[i][j] = Ainv[j][i] = s;
        }
    return true;
}

// ═══════════════════════ prior residual builder ══════════════════════
//
//  Each active prior contributes one "pseudo-observation" to the
//  residual vector:
//
//      r_prior = (X_predicted - X_observed) / σ
//
//  where σ is chosen as σ_lo or σ_hi depending on the sign of the
//  deviation (asymmetric errors).  The geometric sin(i) prior is
//  turned into a soft penalty:  r = -√(w) · ln(sin i).
//
//  The Jacobian rows for these are computed alongside the LC Jacobian
//  via finite differences.
// ═════════════════════════════════════════════════════════════════════

struct PriorResidualSpec {
    string name;
    // callable: given (i, q, vs, r1, r2, t1, t2) → predicted value
    // (we store a tag and switch inside the evaluator)
    enum Tag {
        TAG_K1, TAG_K2, TAG_M1, TAG_M2, TAG_M2MIN, TAG_MTOTAL,
        TAG_Q, TAG_R1_ABS, TAG_R2_ABS, TAG_LOGG1, TAG_LOGG2,
        TAG_T1, TAG_T2, TAG_SINI
    } tag;
    double obs_val;
    double err_lo, err_hi;
    double weight;        // prior_weight multiplier (sqrt applied to residual)
};

static double evaluate_derived(PriorResidualSpec::Tag tag,
                               double i_deg, double q, double vs,
                               double r1, double r2, double t1, double t2,
                               double P_days)
{
    const double DEG2RAD = M_PI / 180.0;
    const double G       = 6.67430e-11;
    const double Msun    = 1.98892e30;
    const double Rsun    = 6.9634e8;

    double sin_i = std::sin(i_deg * DEG2RAD);

    // orbital separation a from velocity scale:  vs = 2π a / P
    double P_s = P_days * 86400.0;
    double a_m = vs * 1e3 * P_s / (2.0 * M_PI);
    double a_rsun = a_m / Rsun;

    // total mass from Kepler III
    double M_total = 4.0 * M_PI * M_PI * a_m * a_m * a_m
                     / (G * P_s * P_s * Msun);
    double M1 = M_total / (1.0 + q);
    double M2 = M_total * q / (1.0 + q);

    // absolute radii
    double R1_abs = r1 * a_rsun;
    double R2_abs = r2 * a_rsun;

    // surface gravities (cgs)
    double g1 = G * M1 * Msun / ((R1_abs * Rsun) * (R1_abs * Rsun));
    double g2 = G * M2 * Msun / ((R2_abs * Rsun) * (R2_abs * Rsun));

    switch (tag) {
        case PriorResidualSpec::TAG_K1:
            return vs * sin_i * q / (1.0 + q);
        case PriorResidualSpec::TAG_K2:
            return vs * sin_i / (1.0 + q);
        case PriorResidualSpec::TAG_M1:
            return M1;
        case PriorResidualSpec::TAG_M2:
            return M2;
        case PriorResidualSpec::TAG_M2MIN: {
            // M2 * sin³i  (minimum mass)
            double sin3 = sin_i * sin_i * sin_i;
            return M2 * sin3;
        }
        case PriorResidualSpec::TAG_MTOTAL:
            return M_total;
        case PriorResidualSpec::TAG_Q:
            return q;
        case PriorResidualSpec::TAG_R1_ABS:
            return R1_abs;
        case PriorResidualSpec::TAG_R2_ABS:
            return R2_abs;
        case PriorResidualSpec::TAG_LOGG1:
            return (g1 > 0) ? std::log10(g1 * 100.0) : 0.0;  // cm/s² → logg
        case PriorResidualSpec::TAG_LOGG2:
            return (g2 > 0) ? std::log10(g2 * 100.0) : 0.0;
        case PriorResidualSpec::TAG_T1:
            return t1;
        case PriorResidualSpec::TAG_T2:
            return t2;
        case PriorResidualSpec::TAG_SINI:
            return sin_i;  // special handling in residual
        default:
            return 0.0;
    }
}

// Build one prior residual: (predicted - observed) / σ, with asymmetric σ
static double prior_residual(const PriorResidualSpec& spec,
                             double i_deg, double q, double vs,
                             double r1, double r2, double t1, double t2,
                             double P_days)
{
    if (spec.tag == PriorResidualSpec::TAG_SINI) {
        // Geometric prior: p(i) ∝ sin(i)
        // ⇒ -ln p(i) = -ln sin(i)
        // Treat as a residual whose square equals 2·(-ln sin i) · weight
        // r² = 2 w ln(1/sin i)  ⇒  r = √(2w) · √(-ln sin i)  with sign
        double sin_i = std::sin(i_deg * M_PI / 180.0);
        if (sin_i <= 0.0) return 1e10;
        double neg_log = -std::log(sin_i);
        return std::sqrt(2.0 * spec.weight * std::max(neg_log, 0.0));
    }

    double predicted = evaluate_derived(spec.tag, i_deg, q, vs,
                                        r1, r2, t1, t2, P_days);
    double diff = predicted - spec.obs_val;
    double sigma = (diff >= 0.0) ? spec.err_hi : spec.err_lo;
    if (sigma <= 0.0) sigma = std::max(spec.err_lo, spec.err_hi);
    if (sigma <= 0.0) sigma = 0.01 * std::abs(spec.obs_val) + 1e-30;

    return std::sqrt(spec.weight) * diff / sigma;
}

// ═════════════════════════════════════════════════════════════════════
//  Parameter helper: read from variable-parameter array or fixed model
// ═════════════════════════════════════════════════════════════════════

static double get_par(int idx, double fixed_val,
                      const Subs::Array1D<double>& pars) {
    return (idx >= 0) ? pars[idx] : fixed_val;
}

// ═════════════════════════════ main ══════════════════════════════════

int main(int argc, char* argv[])
{
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <config_file.json>" << endl;
        return 1;
    }
    ::signal(SIGWINCH, sigwinch_handler);

    // ── Load model & config ──────────────────────────────────────────
    string config_file = argv[1];
    auto model_config = Helpers::load_model_and_config_from_json(config_file);
    Lcurve::Model model = model_config.first;
    json config         = model_config.second;

    // ── Load data ────────────────────────────────────────────────────
    auto data_copy = Helpers::read_and_copy_lightcurve_from_file(
                         config["data_file_path"]);
    Lcurve::Data data = data_copy.first;
    Lcurve::Data copy = data_copy.second;
    bool no_file = data.empty();
    if (no_file) throw Lcurve::Lcurve_Error("No data file provided");
    double noise = config["noise"].get<double>();

    // ── Scale factors ────────────────────────────────────────────────
    int seed;
    bool scale;
    vector<double> sfac;
    Helpers::load_seed_scale_sfac(config, no_file, model, seed, scale, sfac);

    // ── Variable parameters ──────────────────────────────────────────
    int npar = model.nvary();
    vector<string> names(npar);
    for (int i = 0; i < npar; ++i) names[i] = model.get_name(i);

    Subs::Array1D<double> current_pars = model.get_param();
    Subs::Array1D<double> dsteps       = model.get_dstep();
    vector<pair<double,double>> limits  = model.get_limit();
    string device = config.value("plot_device", "none");

    // ── Gnuplot ──────────────────────────────────────────────────────
    Gnuplot gp;
    gp << "set terminal " + device + " title 'Live fitting plot'\n";
    gp << "set grid\n";

    // ── Identify parameter indices for the prior ─────────────────────
    int q_idx = -1, vs_idx = -1, r1_idx = -1, iangle_idx = -1;
    int r2_idx = -1, t1_idx = -1, t2_idx = -1;
    for (int i = 0; i < npar; ++i) {
        if      (names[i] == "q")              q_idx      = i;
        else if (names[i] == "velocity_scale") vs_idx     = i;
        else if (names[i] == "r1")             r1_idx     = i;
        else if (names[i] == "iangle")         iangle_idx = i;
        else if (names[i] == "r2")             r2_idx     = i;
        else if (names[i] == "t1")             t1_idx     = i;
        else if (names[i] == "t2")             t2_idx     = i;
    }

    cout << "Levenberg-Marquardt solver for " << npar << " parameters:" << endl;
    for (int i = 0; i < npar; ++i)
        cout << "  " << names[i] << ": " << current_pars[i]
             << "  step " << dsteps[i]
             << "  limits [" << limits[i].first
             << ", " << limits[i].second << "]" << endl;

    // ─────────────────────────────────────────────────────────────────
    //  LM settings  (MINPACK-compatible defaults)
    //
    //  MINPACK lmder defaults (Moré, Garbow, Hillstrom 1980):
    //    ftol = xtol = √(macheps) ≈ 1.49e-8   (double)
    //    gtol = 0
    //    factor = 100   (initial trust-region radius scale)
    //    maxfev = 200*(n+1)  (max function evaluations)
    //
    //  We use slightly more generous defaults because light-curve
    //  models are expensive and the surface is often noisy.
    // ─────────────────────────────────────────────────────────────────
    const double macheps = std::numeric_limits<double>::epsilon();
    const double sqrt_macheps = std::sqrt(macheps);

    int    max_iter    = config.value("lm_max_iter",    200);
    int    max_fev     = config.value("lm_max_fev",     200 * (npar + 1));
    double ftol        = config.value("lm_ftol",        sqrt_macheps);
    double xtol        = config.value("lm_xtol",        sqrt_macheps);
    double gtol        = config.value("lm_gtol",        0.0);
    double factor      = config.value("lm_factor",      100.0);
    double fd_step_rel = config.value("lm_fd_step_rel", std::sqrt(sqrt_macheps));
    // fd_step_rel ≈ ε^{1/4} ≈ 1.2e-4 is typical for forward differences
    // on a function with O(ε) noise.  MINPACK uses √ε ≈ 1.5e-8 assuming
    // analytically clean functions; for noisy LC models a larger step is
    // safer.  Override in JSON if needed.
    double fd_step_min = config.value("lm_fd_step_min", 1e-10);
    int    max_model_points = config.value("max_model_points", 500);
    int    progress_interval = config.value("progress_interval", 1);
    bool   verbose     = config.value("lm_verbose", true);

    // ─────────────────────────────────────────────────────────────────
    //  Physical priors → augmented residual specifications
    // ─────────────────────────────────────────────────────────────────
    ObservedConstraints obs;
    bool use_priors = config.value("use_priors", false);
    vector<PriorResidualSpec> prior_specs;

    if (use_priors) {
        obs.P_days          = config.value("true_period", 1.0);
        obs.use_sin_i_prior = config.value("use_sin_i_prior", true);
        obs.prior_weight    = config.value("prior_weight", 1.0);
        double pw           = obs.prior_weight;

        for (auto& [p, v] : config["priors"].items()) {
            auto [val, err_lo, err_hi] =
                Helpers::parseThreeDoubles(v.get<string>());
            if (err_hi <= 0.0) err_hi = err_lo;
            if (err_lo <= 0.0) err_lo = err_hi;

            PriorResidualSpec spec;
            spec.obs_val = val;
            spec.err_lo  = err_lo;
            spec.err_hi  = err_hi;
            spec.weight  = pw;

            bool known = true;
            if      (p == "K1" || p == "vrad1_obs") {
                spec.name = "K1"; spec.tag = PriorResidualSpec::TAG_K1;
                obs.K_obs = val; obs.K_err_lo = err_lo;
                obs.K_err_hi = err_hi; obs.has_K = true;
            }
            else if (p == "K2") {
                spec.name = "K2"; spec.tag = PriorResidualSpec::TAG_K2;
                obs.K2_obs = val; obs.K2_err_lo = err_lo;
                obs.K2_err_hi = err_hi; obs.has_K2 = true;
            }
            else if (p == "M1" || p == "m1") {
                spec.name = "M1"; spec.tag = PriorResidualSpec::TAG_M1;
                obs.M1_obs = val; obs.M1_err_lo = err_lo;
                obs.M1_err_hi = err_hi; obs.has_M1 = true;
            }
            else if (p == "M2") {
                spec.name = "M2"; spec.tag = PriorResidualSpec::TAG_M2;
                obs.M2_obs = val; obs.M2_err_lo = err_lo;
                obs.M2_err_hi = err_hi; obs.has_M2 = true;
            }
            else if (p == "M2_min" || p == "m2_min") {
                spec.name = "M2min"; spec.tag = PriorResidualSpec::TAG_M2MIN;
                obs.M2min_obs = val; obs.M2min_err_lo = err_lo;
                obs.M2min_err_hi = err_hi; obs.has_M2min = true;
            }
            else if (p == "M_total") {
                spec.name = "M_total"; spec.tag = PriorResidualSpec::TAG_MTOTAL;
                obs.Mtotal_obs = val; obs.Mtotal_err_lo = err_lo;
                obs.Mtotal_err_hi = err_hi; obs.has_Mtotal = true;
            }
            else if (p == "q") {
                spec.name = "q"; spec.tag = PriorResidualSpec::TAG_Q;
                obs.q_obs = val; obs.q_err_lo = err_lo;
                obs.q_err_hi = err_hi; obs.has_q = true;
            }
            else if (p == "R1" || p == "r1") {
                spec.name = "R1"; spec.tag = PriorResidualSpec::TAG_R1_ABS;
                obs.R1_obs = val; obs.R1_err_lo = err_lo;
                obs.R1_err_hi = err_hi; obs.has_R1 = true;
            }
            else if (p == "R2") {
                spec.name = "R2"; spec.tag = PriorResidualSpec::TAG_R2_ABS;
                obs.R2_obs = val; obs.R2_err_lo = err_lo;
                obs.R2_err_hi = err_hi; obs.has_R2 = true;
            }
            else if (p == "logg1") {
                spec.name = "logg1"; spec.tag = PriorResidualSpec::TAG_LOGG1;
                obs.logg1_obs = val; obs.logg1_err_lo = err_lo;
                obs.logg1_err_hi = err_hi; obs.has_logg1 = true;
            }
            else if (p == "logg2") {
                spec.name = "logg2"; spec.tag = PriorResidualSpec::TAG_LOGG2;
                obs.logg2_obs = val; obs.logg2_err_lo = err_lo;
                obs.logg2_err_hi = err_hi; obs.has_logg2 = true;
            }
            else if (p == "T1") {
                spec.name = "T1"; spec.tag = PriorResidualSpec::TAG_T1;
                obs.T1_obs = val; obs.T1_err_lo = err_lo;
                obs.T1_err_hi = err_hi; obs.has_T1 = true;
            }
            else if (p == "T2") {
                spec.name = "T2"; spec.tag = PriorResidualSpec::TAG_T2;
                obs.T2_obs = val; obs.T2_err_lo = err_lo;
                obs.T2_err_hi = err_hi; obs.has_T2 = true;
            }
            else { cerr << "Unknown prior: " << p << endl; return 1; }

            if (known) prior_specs.push_back(spec);
        }

        // Print active priors
        cout << BRIGHT_CYAN << "Physical priors (P = "
             << obs.P_days << " d):" << RESET << endl;
        for (auto& sp : prior_specs) {
            if (sp.tag == PriorResidualSpec::TAG_SINI)
                cout << "  p(i) ~ sin(i)  (geometric prior)\n";
            else
                cout << "  " << sp.name << " = " << sp.obs_val
                     << " ± " << sp.err_lo << "/" << sp.err_hi
                     << "  (weight × " << sp.weight << ")\n";
        }
    }

    const int ndata   = static_cast<int>(data.size());
    const int nprior  = static_cast<int>(prior_specs.size());
    const int nresid  = ndata + nprior;

    cout << "Residual vector: " << ndata << " data + "
         << nprior << " prior = " << nresid << " total" << endl;

    if (nresid < npar) {
        cerr << BRIGHT_RED << "ERROR: under-determined system ("
             << nresid << " residuals < " << npar << " parameters)"
             << RESET << endl;
        return 1;
    }

    // ─────────────────────────────────────────────────────────────────
    //  Auto-adjust starting parameters (same logic as MCMC solver)
    // ─────────────────────────────────────────────────────────────────
    bool auto_init = config.value("auto_consistent_init", true);
    if (use_priors && auto_init && obs.has_K && obs.has_M1 && obs.has_R1)
    {
        double init_i  = get_par(iangle_idx, model.iangle.value, current_pars);
        double init_q  = get_par(q_idx,      model.q.value,      current_pars);
        double init_vs = get_par(vs_idx,     model.velocity_scale.value, current_pars);
        double init_r1 = get_par(r1_idx,     model.r1.value,     current_pars);

        double sin_i_init = std::sin(init_i * M_PI / 180.0);
        double K_implied  = init_vs * sin_i_init * init_q / (1.0 + init_q);

        double K_sigma = (K_implied < obs.K_obs) ? obs.K_err_lo : obs.K_err_hi;
        if (K_sigma <= 0.0) K_sigma = std::max(obs.K_err_lo, obs.K_err_hi);
        if (K_sigma <= 0.0) K_sigma = 0.1 * obs.K_obs;

        double K_pull = (K_implied - obs.K_obs) / K_sigma;

        if (std::abs(K_pull) > 3.0) {
            cout << BRIGHT_YELLOW
                 << "\n  Starting K1 = " << K_implied
                 << " is " << std::abs(K_pull) << "σ from K_obs = "
                 << obs.K_obs << "\n  Computing consistent starting parameters..."
                 << RESET << endl;

            double q_new, vs_new, r1_new;
            bool ok = PhysicalPrior::solve_consistent_params(
                          init_i, obs.K_obs, obs.M1_obs, obs.R1_obs,
                          obs.P_days, q_new, vs_new, r1_new);

            if (!ok) {
                for (double try_i = 85.0; try_i >= 30.0; try_i -= 5.0) {
                    ok = PhysicalPrior::solve_consistent_params(
                             try_i, obs.K_obs, obs.M1_obs, obs.R1_obs,
                             obs.P_days, q_new, vs_new, r1_new);
                    if (ok) { init_i = try_i; break; }
                }
            }

            if (ok) {
                cout << BRIGHT_GREEN
                     << "  Consistent starting point found:" << RESET << endl;
                cout << "    i  = " << init_i << " deg\n"
                     << "    q  = " << q_new  << "\n"
                     << "    vs = " << vs_new << " km/s\n"
                     << "    r1 = " << r1_new << endl;

                if (iangle_idx >= 0) current_pars[iangle_idx] = init_i;
                if (q_idx >= 0)      current_pars[q_idx]      = q_new;
                if (vs_idx >= 0)     current_pars[vs_idx]     = vs_new;
                if (r1_idx >= 0)     current_pars[r1_idx]     = r1_new;

                for (int i = 0; i < npar; ++i)
                    current_pars[i] = std::clamp(current_pars[i],
                                                 limits[i].first,
                                                 limits[i].second);
                model.set_param(current_pars);
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────
    //  Lambda: evaluate full residual vector at given parameters
    //
    //  prior_scale ∈ [0,1] controls how strongly priors contribute.
    //  During continuation this ramps from 0 → 1 so that the LC
    //  fit is established before priors pull the solution around.
    // ─────────────────────────────────────────────────────────────────
    int fev_count = 0;
    
    // ─────────────────────────────────────────────────────────────────
    //  Prior balancing
    //
    //  The raw prior_weight can easily overwhelm a well-fitting LC
    //  (χ²/N ≪ 1).  We normalise so that the total prior penalty at
    //  1σ tension equals a user-controllable fraction of the current
    //  LC χ².  This keeps LM from abandoning the LC fit.
    //
    //  effective_prior_scale = active_prior_scale × balance_factor
    //  where balance_factor = (χ²_LC / N_prior) / (prior_weight)
    //  when auto-balance is on, or 1.0 when off.
    // ─────────────────────────────────────────────────────────────────
    double active_prior_scale = 0.0;
    double prior_balance_factor = 1.0;
    bool   auto_balance_priors = config.value("lm_auto_balance_priors", true);
    // Fraction of LC chi² that total prior at 1σ should equal
    double prior_balance_target = config.value("lm_prior_balance_target", 1.0);

    auto compute_residuals = [&](const Subs::Array1D<double>& pars,
                                 vector<double>& resid,
                                 double& chisq_lc,
                                 vector<double>& fit_out) -> bool
    {
        model.set_param(pars);

        vector<double> fitv;
        double wd, ch, wn, lg1, lg2, rv1, rv2;
        try {
            light_curve_comp_fast(model, data, scale, !no_file, false, sfac,
                                 fitv, wd, ch, wn, lg1, lg2, rv1, rv2,
                                 max_model_points);
        } catch (Lcurve::Lcurve_Error&) {
            return false;
        }
        ++fev_count;

        resid.resize(nresid);
        chisq_lc = 0.0;

        // Data residuals: (data - model) / σ
        for (int i = 0; i < ndata; ++i) {
            double w = data[i].weight;
            double sigma = (w > 0.0) ? 1.0 / std::sqrt(w) : 1e10;
            resid[i] = (data[i].flux - fitv[i]) / sigma;
            chisq_lc += resid[i] * resid[i];
        }


        // Prior residuals — scaled by continuation weight and balance
        if (use_priors) {
            double pi  = get_par(iangle_idx, model.iangle.value, pars);
            double pq  = get_par(q_idx,      model.q.value,      pars);
            double pv  = get_par(vs_idx,     model.velocity_scale.value, pars);
            double pr  = get_par(r1_idx,     model.r1.value,     pars);
            double pr2 = get_par(r2_idx,     model.r2.value,     pars);
            double pt1 = get_par(t1_idx,     model.t1.value,     pars);
            double pt2 = get_par(t2_idx,     model.t2.value,     pars);

            // effective scale = continuation ramp × balance factor
            // sqrt because residuals are squared in the cost
            double eff = std::sqrt(active_prior_scale * prior_balance_factor);

            for (int k = 0; k < nprior; ++k) {
                resid[ndata + k] = eff * prior_residual(
                    prior_specs[k], pi, pq, pv, pr, pr2, pt1, pt2,
                    obs.P_days);
            }
        }

        fit_out = std::move(fitv);
        return true;
    };

    // ─────────────────────────────────────────────────────────────────
    //  Lambda: compute Jacobian via forward finite differences
    //
    //  MINPACK convention: J[i][j] = ∂r_i / ∂p_j
    //  Step size:  h_j = fd_step_rel · |p_j| + fd_step_min
    //  (MINPACK uses √ε · |p_j|; we default to ε^{1/4} for robustness)
    //
    //  If a step would push p_j outside its bounds, we use a backward
    //  difference instead.
    // ─────────────────────────────────────────────────────────────────
    auto compute_jacobian = [&](const Subs::Array1D<double>& pars,
                                const vector<double>& resid0,
                                vector<vector<double>>& J) -> bool
    {
        J.assign(nresid, vector<double>(npar, 0.0));

        for (int j = 0; j < npar; ++j) {
            double hj = fd_step_rel * std::abs(pars[j]) + fd_step_min;

            // Respect bounds: prefer forward, fall back to backward
            double pj_pert = pars[j] + hj;
            double sign = 1.0;

            if (pj_pert > limits[j].second) {
                // backward difference
                pj_pert = pars[j] - hj;
                sign = -1.0;
                if (pj_pert < limits[j].first) {
                    // squeezed between bounds — use central with tiny step
                    hj = 0.5 * (limits[j].second - limits[j].first);
                    if (hj < 1e-15) {
                        // parameter is essentially fixed by bounds
                        continue;
                    }
                    pj_pert = pars[j] + hj;
                    sign = 1.0;
                }
            }

            Subs::Array1D<double> p_pert = pars;
            p_pert[j] = pj_pert;
            double actual_h = sign * (p_pert[j] - pars[j]);

            vector<double> resid_pert;
            double chisq_dummy;
            vector<double> fit_dummy;
            bool ok = compute_residuals(p_pert, resid_pert,
                                        chisq_dummy, fit_dummy);
            if (!ok) {
                // Try backward if forward failed
                if (sign > 0) {
                    p_pert[j] = pars[j] - hj;
                    actual_h = -(pars[j] - p_pert[j]);
                    ok = compute_residuals(p_pert, resid_pert,
                                           chisq_dummy, fit_dummy);
                }
                if (!ok) continue;  // leave column as zeros
            }

            double inv_h = 1.0 / actual_h;
            for (int i = 0; i < nresid; ++i)
                J[i][j] = (resid_pert[i] - resid0[i]) * inv_h;
        }
        return true;
    };

    // ─────────────────────────────────────────────────────────────────
    //  Initial evaluation
    // ─────────────────────────────────────────────────────────────────
    vector<double> resid(nresid);
    double chisq_lc;
    vector<double> current_fit;
    {
        bool ok = compute_residuals(current_pars, resid, chisq_lc, current_fit);
        if (!ok) {
            cerr << "Initial parameter evaluation failed!" << endl;
            return 1;
        }
    }

    double sum_sq = 0.0;
    for (double r : resid) sum_sq += r * r;

    cout << "\nInitial  χ²(LC) = " << fixed << setprecision(4) << chisq_lc
         << "   total ‖r‖² = " << sum_sq << endl;

    if (use_priors) {
        double prior_sq = 0.0;
        for (int k = 0; k < nprior; ++k)
            prior_sq += resid[ndata + k] * resid[ndata + k];
        cout << "  Prior contribution to ‖r‖²: " << prior_sq << endl;

        double init_i  = get_par(iangle_idx, model.iangle.value, current_pars);
        double init_q  = get_par(q_idx,      model.q.value,      current_pars);
        double init_vs = get_par(vs_idx,     model.velocity_scale.value, current_pars);
        double init_r1 = get_par(r1_idx,     model.r1.value,     current_pars);
        double init_r2 = get_par(r2_idx,     model.r2.value,     current_pars);
        double init_t1 = get_par(t1_idx,     model.t1.value,     current_pars);
        double init_t2 = get_par(t2_idx,     model.t2.value,     current_pars);
        cout << BRIGHT_CYAN << "Starting implied quantities:" << RESET << endl;
        PhysicalPrior::print_implied(init_i, init_q, init_vs, init_r1,
                                     init_r2, init_t1, init_t2, obs);
    }

    // ─────────────────────────────────────────────────────────────────
    //  MINPACK-style diagonal scaling
    //
    //  D[j] = max over all iterations of ‖column j of J‖
    //  This is initialised from the first Jacobian and updated each
    //  iteration (MINPACK mode=1).
    // ─────────────────────────────────────────────────────────────────
    vector<double> D(npar, 1.0);
    bool D_initialised = false;

    auto update_scaling = [&](const vector<vector<double>>& J)
    {
        for (int j = 0; j < npar; ++j) {
            double col_norm = 0.0;
            for (int i = 0; i < nresid; ++i)
                col_norm += J[i][j] * J[i][j];
            col_norm = std::sqrt(col_norm);

            if (!D_initialised)
                D[j] = (col_norm > 0.0) ? col_norm : 1.0;
            else
                D[j] = std::max(D[j], col_norm);
        }
        D_initialised = true;
    };

    // ─────────────────────────────────────────────────────────────────
    //  Compute J^T J  and  J^T r
    // ─────────────────────────────────────────────────────────────────
    auto compute_JtJ_Jtr = [&](const vector<vector<double>>& J,
                                const vector<double>& r,
                                vector<vector<double>>& JtJ,
                                vector<double>& Jtr)
    {
        JtJ.assign(npar, vector<double>(npar, 0.0));
        Jtr.assign(npar, 0.0);

        for (int j = 0; j < npar; ++j) {
            for (int k = j; k < npar; ++k) {
                double s = 0.0;
                for (int i = 0; i < nresid; ++i)
                    s += J[i][j] * J[i][k];
                JtJ[j][k] = JtJ[k][j] = s;
            }
            for (int i = 0; i < nresid; ++i)
                Jtr[j] += J[i][j] * r[i];
        }
    };

    // ═════════════════════════════════════════════════════════════════
    //  PRIOR CONTINUATION SCHEDULE
    //
    //  Phase 1: fit LC only (prior_scale = 0)
    //  Phase 2: ramp prior_scale from 0 → 1 in steps
    //  Phase 3: final polish at full prior weight
    //
    //  This prevents the optimizer from ignoring the light curve
    //  (which has tiny χ² gradients for non-eclipsing systems) in
    //  favour of chasing the much steeper prior terms.
    // ═════════════════════════════════════════════════════════════════
    bool   continuation_enabled = config.value("lm_continuation", true);
    int    continuation_stages  = config.value("lm_continuation_stages", 6);
    // stages = number of steps from 0→1 inclusive of endpoints
    // e.g. 6 means: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0

    vector<double> continuation_schedule;
    if (continuation_enabled && use_priors && nprior > 0) {
        for (int s = 0; s <= continuation_stages; ++s)
            continuation_schedule.push_back(
                double(s) / continuation_stages);
    } else {
        continuation_schedule.push_back(1.0);  // single stage, full weight
    }

    auto t_start = Clock::now();

    // ── Track best solution across all stages ────────────────────────
    Subs::Array1D<double> best_pars = current_pars;
    double best_sum_sq  = 1e30;
    double best_chisq   = 1e30;
    vector<double> best_fit = current_fit;

    struct IterLog {
        int    iter;
        double sum_sq;
        double chisq_lc;
        double lambda;
        double step_norm;
        double gain_ratio;
        int    fev;
        double prior_scale;
        string status;
    };
    vector<IterLog> iter_log;

    int  total_iter = 0;
    bool converged  = false;
    string stop_reason = "max_iter";
    double lambda = -1.0;  // will be initialised per stage

    // ═════════════════════════════════════════════════════════════════
    //  CONTINUATION OUTER LOOP
    // ═════════════════════════════════════════════════════════════════
    for (size_t stage = 0; stage < continuation_schedule.size(); ++stage)
    {
        active_prior_scale = continuation_schedule[stage];
        // ── Compute prior balance factor for this stage ──────────────
        //
        //  Evaluate LC-only chi² at current parameters to set the
        //  scale.  Goal: at 1σ tension on every prior, the total
        //  prior contribution ≈ prior_balance_target × χ²_LC.
        //
        //  Without this, prior_weight=28.57 with χ²_LC=0.16 means
        //  one prior at 1σ contributes 28.57 while the ENTIRE LC
        //  contributes 0.16. LM will ignore the light curve.
        //
        if (auto_balance_priors && use_priors && nprior > 0
            && active_prior_scale > 0.0)
        {
            // Get LC chi² with priors off
            double saved_ps = active_prior_scale;
            active_prior_scale = 0.0;
            vector<double> r_lc;
            double chisq_lc_only;
            vector<double> f_tmp;
            compute_residuals(current_pars, r_lc, chisq_lc_only, f_tmp);
            active_prior_scale = saved_ps;

            // At 1σ on all priors, raw total = nprior × prior_weight × 1²
            // We want: nprior × prior_weight × balance = target × χ²_LC
            // ⇒ balance = target × χ²_LC / (nprior × prior_weight)
            double raw_prior_per_sigma = 0.0;
            for (auto& sp : prior_specs)
                raw_prior_per_sigma += sp.weight;  // weight × 1²

            if (raw_prior_per_sigma > 0.0 && chisq_lc_only > 1e-30) {
                prior_balance_factor = prior_balance_target * chisq_lc_only
                                     / raw_prior_per_sigma;
                // Don't let balance amplify beyond original weight
                prior_balance_factor = std::min(prior_balance_factor, 1.0);
            } else {
                prior_balance_factor = 1.0;
            }

            if (verbose) {
                cout << "  Prior balance: χ²(LC) = " << fixed
                     << setprecision(4) << chisq_lc_only
                     << "  raw_prior_per_1σ = " << setprecision(2)
                     << raw_prior_per_sigma
                     << "  balance_factor = " << scientific
                     << setprecision(3) << prior_balance_factor
                     << RESET << endl;
                if (prior_balance_factor < 0.01) {
                    cout << BRIGHT_YELLOW
                         << "  Prior weight heavily reduced to protect"
                         << " LC fit (χ² very small vs prior penalty)."
                         << "\n  Each prior at 1σ now contributes ~"
                         << fixed << setprecision(4)
                         << (prior_balance_factor
                             * prior_specs[0].weight)
                         << " to ‖r‖² (was "
                         << prior_specs[0].weight << ")"
                         << RESET << endl;
                }
            }
        }
        else {
            prior_balance_factor = (active_prior_scale > 0.0) ? 1.0 : 0.0;
        }
        bool is_final_stage = (stage == continuation_schedule.size() - 1);

        // Per-stage iteration limit: more generous for final stage
        int stage_max_iter = is_final_stage ? max_iter
                           : max(20, max_iter / (int)continuation_schedule.size());
        // Per-stage convergence: looser for intermediate stages
        double stage_ftol = is_final_stage ? ftol : std::max(ftol, 1e-4);
        double stage_xtol = is_final_stage ? xtol : std::max(xtol, 1e-4);

        if (continuation_schedule.size() > 1) {
            cout << "\n" << BRIGHT_CYAN
                 << "─── Continuation stage " << stage + 1 << "/"
                 << continuation_schedule.size()
                 << "  prior_scale = " << fixed << setprecision(2)
                 << active_prior_scale << " ───"
                 << RESET << endl;
        }

        // ── Evaluate residuals at current point for this stage ───────
        vector<double> resid(nresid);
        double chisq_lc;
        {
            bool ok = compute_residuals(current_pars, resid, chisq_lc,
                                        current_fit);
            if (!ok) {
                cerr << "Evaluation failed at continuation stage "
                     << stage << endl;
                break;
            }
        }
        double sum_sq = 0.0;
        for (double r : resid) sum_sq += r * r;

        if (verbose) {
            cout << "  Start:  χ²(LC) = " << fixed << setprecision(4)
                 << chisq_lc << "   ‖r‖² = " << sum_sq;
            if (active_prior_scale > 0) {
                double psq = 0;
                for (int k = 0; k < nprior; ++k)
                    psq += resid[ndata + k] * resid[ndata + k];
                cout << "  (prior contrib: " << psq << ")";
            }
            cout << endl;
        }

        // ── Jacobian ────────────────────────────────────────────────
        vector<vector<double>> J;
        compute_jacobian(current_pars, resid, J);
        update_scaling(J);

        vector<vector<double>> JtJ;
        vector<double> Jtr;
        compute_JtJ_Jtr(J, resid, JtJ, Jtr);

        // ── λ initialisation ─────────────────────────────────────────
        //  Re-initialise at first stage or when prior weight changes
        //  significantly, because the Hessian landscape has shifted.
        if (lambda < 0.0 || stage > 0) {
            double max_diag = 0.0;
            for (int j = 0; j < npar; ++j)
                max_diag = std::max(max_diag, JtJ[j][j]);
            double tau = config.value("lm_tau", 1e-3);
            lambda = tau * max(max_diag, 1e-30);
        }
        double nu = 2.0;

        if (verbose) {
            cout << "  " << setw(5) << left << "Iter"
                 << setw(14) << right << "‖r‖²"
                 << setw(14) << "χ²(LC)"
                 << setw(12) << "λ"
                 << setw(12) << "‖δ‖"
                 << setw(10) << "ρ"
                 << setw(6)  << "FEV"
                 << "  Status" << endl;
            cout << string(80, '-') << endl;
        }

        int  stage_iter = 0;
        int  consecutive_fails = 0;
        bool stage_converged = false;

        // ═════════════════════════════════════════════════════════════
        //  LM INNER LOOP (per continuation stage)
        // ═════════════════════════════════════════════════════════════
        for (stage_iter = 0; stage_iter < stage_max_iter; ++stage_iter)
        {
            ++total_iter;

            // ── Check gradient convergence ───────────────────────────
            if (gtol > 0.0) {
                double rnorm = std::sqrt(sum_sq);
                double gmax  = 0.0;
                for (int j = 0; j < npar; ++j)
                    gmax = std::max(gmax,
                        std::abs(Jtr[j]) / std::max(1.0, D[j]));
                if (gmax / std::max(1.0, rnorm) < gtol) {
                    stage_converged = true;
                    stop_reason = "gtol (gradient small)";
                    break;
                }
            }

            // ── Solve (J^T J + λ D²) δ = -J^T r ────────────────────
            vector<vector<double>> A = JtJ;
            vector<double> delta;
            bool solve_ok = solve_normal_equations(A, Jtr, D, lambda, delta);

            if (!solve_ok) {
                lambda *= nu;
                nu *= 2.0;
                consecutive_fails++;
                if (verbose)
                    cout << "  " << setw(5) << left << stage_iter
                         << "  Cholesky failed, λ → "
                         << scientific << lambda << endl;
                if (consecutive_fails > 50) {
                    stop_reason = "singular (Cholesky failures)";
                    break;
                }
                continue;
            }

            // ── Clamp step to bounds ─────────────────────────────────
            Subs::Array1D<double> trial = current_pars;
            double step_norm_sq = 0.0;
            for (int j = 0; j < npar; ++j) {
                trial[j] = current_pars[j] + delta[j];
                trial[j] = std::clamp(trial[j],
                                      limits[j].first, limits[j].second);
                delta[j] = trial[j] - current_pars[j];
                step_norm_sq += (delta[j] * D[j]) * (delta[j] * D[j]);
            }
            double step_norm = std::sqrt(step_norm_sq);

            // ── xtol check ───────────────────────────────────────────
            {
                double xnorm_sq = 0.0;
                for (int j = 0; j < npar; ++j)
                    xnorm_sq += (current_pars[j] * D[j])
                               * (current_pars[j] * D[j]);
                double xnorm = std::sqrt(xnorm_sq);
                if (xnorm > 0.0 && step_norm / xnorm < stage_xtol) {
                    stage_converged = true;
                    if (is_final_stage)
                        stop_reason = "xtol (parameter change small)";
                    break;
                }
            }

            // ── Evaluate trial ───────────────────────────────────────
            vector<double> resid_trial;
            double chisq_lc_trial;
            vector<double> fit_trial;
            bool eval_ok = compute_residuals(trial, resid_trial,
                                             chisq_lc_trial, fit_trial);

            if (!eval_ok) {
                lambda *= nu;
                nu *= 2.0;
                consecutive_fails++;
                iter_log.push_back({total_iter, sum_sq, chisq_lc, lambda,
                    step_norm, -1.0, fev_count, active_prior_scale,
                    "eval_fail"});
                if (consecutive_fails > 50) {
                    stop_reason = "max consecutive failures";
                    break;
                }
                continue;
            }

            double sum_sq_trial = 0.0;
            for (double r : resid_trial) sum_sq_trial += r * r;

            // ── Gain ratio ───────────────────────────────────────────
            double Jdelta_norm_sq = 0.0;
            for (int i = 0; i < nresid; ++i) {
                double Jd = 0.0;
                for (int j = 0; j < npar; ++j)
                    Jd += J[i][j] * delta[j];
                Jdelta_norm_sq += Jd * Jd;
            }
            double Ddelta_norm_sq = 0.0;
            for (int j = 0; j < npar; ++j)
                Ddelta_norm_sq += (D[j]*delta[j]) * (D[j]*delta[j]);

            double predicted = Jdelta_norm_sq
                             + 2.0 * lambda * Ddelta_norm_sq;
            double actual    = sum_sq - sum_sq_trial;
            double rho = (predicted > 0.0) ? actual / predicted : 0.0;

            string status;

            if (rho > 0.0) {
                current_pars = trial;
                resid        = resid_trial;
                sum_sq       = sum_sq_trial;
                chisq_lc     = chisq_lc_trial;
                current_fit  = fit_trial;

                if (is_final_stage && sum_sq < best_sum_sq) {
                    best_sum_sq = sum_sq;
                    best_chisq  = chisq_lc;
                    best_pars   = current_pars;
                    best_fit    = current_fit;
                }

                double tmp = 2.0 * rho - 1.0;
                lambda *= std::max(1.0/3.0, 1.0 - tmp*tmp*tmp);
                nu = 2.0;

                compute_jacobian(current_pars, resid, J);
                update_scaling(J);
                compute_JtJ_Jtr(J, resid, JtJ, Jtr);

                consecutive_fails = 0;
                status = "accept";

                // ftol check
                if (sum_sq > 0.0) {
                    double rel_actual = std::abs(actual) / sum_sq;
                    double rel_pred   = predicted / sum_sq;
                    if (rel_actual < stage_ftol && rel_pred < stage_ftol) {
                        stage_converged = true;
                        if (is_final_stage)
                            stop_reason = "ftol (cost converged)";
                        status = "ftol";
                    }
                }
            } else {
                lambda *= nu;
                nu *= 2.0;
                consecutive_fails++;
                status = "reject";
            }

            lambda = std::clamp(lambda, 1e-20, 1e20);

            iter_log.push_back({total_iter, sum_sq, chisq_lc, lambda,
                step_norm, rho, fev_count, active_prior_scale, status});

            if (verbose && (stage_iter % progress_interval == 0
                            || stage_converged))
            {
                string col;
                if      (status == "accept" || status == "ftol")
                    col = BRIGHT_GREEN;
                else if (status == "reject") col = BRIGHT_YELLOW;
                else                         col = BRIGHT_RED;

                cout << "  " << setw(5) << left << stage_iter
                     << setw(14) << right << fixed << setprecision(4)
                     << sum_sq
                     << setw(14) << chisq_lc
                     << setw(12) << scientific << setprecision(2) << lambda
                     << setw(12) << setprecision(2) << step_norm
                     << setw(10) << fixed << setprecision(4) << rho
                     << setw(6) << fev_count
                     << "  " << col << status << RESET << endl;
            }

            if (stage_iter % max(1, progress_interval) == 0)
                Helpers::plot_model_live(data, current_fit, no_file,
                                        copy, gp);

            if (stage_converged) break;
            if (fev_count >= max_fev) {
                stop_reason = "max function evaluations";
                goto lm_done;  // break out of both loops
            }
        } // ═══ end LM inner loop ═══

        if (is_final_stage) {
            converged = stage_converged;
            if (!stage_converged && stop_reason == "max_iter")
                stop_reason = "max_iter (final stage)";
        }

        // ── If LC-only stage found a non-eclipse solution, record it ─
        if (stage == 0 && !use_priors) {
            best_sum_sq = sum_sq;
            best_chisq  = chisq_lc;
            best_pars   = current_pars;
            best_fit    = current_fit;
        }

    } // ═══ end continuation outer loop ═══
    lm_done:

    // Ensure best is set if continuation had only one stage
    if (best_sum_sq > 1e29) {
        double ss = 0;
        vector<double> r_tmp;
        double ch_tmp;
        vector<double> f_tmp;
        active_prior_scale = 1.0;
        compute_residuals(current_pars, r_tmp, ch_tmp, f_tmp);
        for (double rr : r_tmp) ss += rr * rr;
        best_sum_sq = ss;
        best_chisq  = ch_tmp;
        best_pars   = current_pars;
        best_fit    = f_tmp;
    }

    // ─────────────────────────────────────────────────────────────────
    //  Timing
    // ─────────────────────────────────────────────────────────────────
    auto t_end  = Clock::now();
    auto dur_ms = chrono::duration_cast<chrono::milliseconds>(
                      t_end - t_start).count();
    double total_s = dur_ms / 1000.0;
    int hrs  = int(total_s / 3600);
    int mins = int((total_s - hrs * 3600) / 60);
    double secs = total_s - hrs * 3600 - mins * 60;

    cout << "\n" << BRIGHT_CYAN
         << "─── LM iteration complete ──────────────────────────"
         << RESET << endl;
    cout << "  Stop reason: " << (converged ? BRIGHT_GREEN : BRIGHT_YELLOW)
         << stop_reason << RESET << endl;
    cout << "  Iterations:  " << total_iter << " / " << max_iter << endl;
    cout << "  Func evals:  " << fev_count << " / " << max_fev << endl;
    cout << "  Time:        ";
    if (hrs  > 0) cout << hrs  << "h ";
    if (mins > 0) cout << mins << "m ";
    cout << fixed << setprecision(2) << secs << "s" << endl;

    // ─────────────────────────────────────────────────────────────────
    //  Best-fit report
    // ─────────────────────────────────────────────────────────────────
    cout << "\n" << BRIGHT_GREEN << "Best solution:" << RESET << endl;
    cout << "  χ²(LC) = " << best_chisq
         << "   total ‖r‖² = " << best_sum_sq << endl;
    cout << "  reduced χ² = " << best_chisq / max(1, ndata - npar) << endl;

    for (int i = 0; i < npar; ++i)
        cout << "  " << names[i] << " = " << best_pars[i] << endl;

    if (use_priors) {
        double bi  = get_par(iangle_idx, model.iangle.value, best_pars);
        double bq  = get_par(q_idx,      model.q.value,      best_pars);
        double bv  = get_par(vs_idx,     model.velocity_scale.value, best_pars);
        double br  = get_par(r1_idx,     model.r1.value,     best_pars);
        double br2 = get_par(r2_idx,     model.r2.value,     best_pars);
        double bt1 = get_par(t1_idx,     model.t1.value,     best_pars);
        double bt2 = get_par(t2_idx,     model.t2.value,     best_pars);

        cout << BRIGHT_CYAN << "Best-fit implied quantities:" << RESET << endl;
        PhysicalPrior::print_implied(bi, bq, bv, br, br2, bt1, bt2, obs);

        // Individual prior residuals
        cout << BRIGHT_CYAN << "Individual prior residuals:" << RESET << endl;
        vector<double> best_resid;
        double best_chisq_lc;
        vector<double> best_fit_final;
        compute_residuals(best_pars, best_resid, best_chisq_lc, best_fit_final);
        for (int k = 0; k < nprior; ++k) {
            double r = best_resid[ndata + k];
            string col = (std::abs(r) < 2.0) ? BRIGHT_GREEN
                       : (std::abs(r) < 3.0) ? BRIGHT_YELLOW
                       : BRIGHT_RED;
            cout << "  " << setw(10) << left << prior_specs[k].name
                 << "  r = " << col << fixed << setprecision(3)
                 << r << RESET << "  (|r| = " << std::abs(r) << "σ)" << endl;
        }
    }

    // ─────────────────────────────────────────────────────────────────
    //  Covariance matrix and formal uncertainties
    //
    //  At the solution, the parameter covariance is approximately:
    //    C = s² · (J^T J)^{-1}
    //  where s² = ‖r‖² / (m - n)   (residual variance)
    //
    //  If priors are included, they are part of the residual, so the
    //  covariance naturally reflects the prior constraints.
    // ─────────────────────────────────────────────────────────────────
    {
        model.set_param(best_pars);
        vector<double> best_resid_final;
        double best_chisq_lc_final;
        vector<double> dummy_fit;
        compute_residuals(best_pars, best_resid_final,
                          best_chisq_lc_final, dummy_fit);

        vector<vector<double>> J_final;
        compute_jacobian(best_pars, best_resid_final, J_final);

        vector<vector<double>> JtJ_final;
        vector<double> Jtr_final;
        compute_JtJ_Jtr(J_final, best_resid_final, JtJ_final, Jtr_final);

        double s2 = best_sum_sq / std::max(1, nresid - npar);

        vector<vector<double>> cov;
        bool inv_ok = invert_spd(JtJ_final, cov);

        if (inv_ok) {
            // Scale by s²
            for (int i = 0; i < npar; ++i)
                for (int j = 0; j < npar; ++j)
                    cov[i][j] *= s2;

            cout << "\n" << BRIGHT_CYAN
                 << "Parameter uncertainties (1σ, from covariance):"
                 << RESET << endl;

            size_t mw = 0;
            for (int i = 0; i < npar; ++i)
                mw = std::max(mw, names[i].size());

            vector<double> sigma(npar);
            for (int i = 0; i < npar; ++i) {
                sigma[i] = std::sqrt(std::max(cov[i][i], 0.0));
                cout << "  " << setw(int(mw)) << left << names[i]
                     << " = " << fixed << setprecision(6) << best_pars[i]
                     << " ± " << sigma[i] << endl;
            }

            // Correlation matrix
            if (npar <= 20) {
                cout << "\n" << BRIGHT_CYAN << "Correlation matrix:"
                     << RESET << endl;
                const int cw = std::max(8, int(mw) + 1);
                cout << "  " << setw(int(mw)) << " ";
                for (int j = 0; j < npar; ++j)
                    cout << setw(cw) << right
                         << names[j].substr(0, cw - 1);
                cout << endl;

                for (int i = 0; i < npar; ++i) {
                    cout << "  " << setw(int(mw)) << left << names[i];
                    for (int j = 0; j < npar; ++j) {
                        double r = (sigma[i] > 0 && sigma[j] > 0)
                            ? cov[i][j] / (sigma[i] * sigma[j])
                            : 0.0;
                        string col = RESET;
                        if (i == j)           col = DIM;
                        else if (abs(r) > .7) col = BRIGHT_RED;
                        else if (abs(r) > .4) col = BRIGHT_YELLOW;
                        cout << col << setw(cw) << right
                             << fixed << setprecision(2) << r << RESET;
                    }
                    cout << endl;
                }
            }

            // Store in config for output
            json cov_info;
            for (int i = 0; i < npar; ++i) {
                cov_info["sigma"][names[i]]      = sigma[i];
                cov_info["best_fit"][names[i]]   = (double)best_pars[i];
                for (int j = 0; j < npar; ++j) {
                    string key = names[i] + "," + names[j];
                    cov_info["covariance"][key]   = cov[i][j];
                    double r = (sigma[i] > 0 && sigma[j] > 0)
                        ? cov[i][j] / (sigma[i] * sigma[j]) : 0.0;
                    cov_info["correlations"][key] = r;
                }
            }
            cov_info["reduced_chi2"] = best_chisq_lc_final / max(1, ndata - npar);
            cov_info["residual_variance"] = s2;
            config["lm_results"] = cov_info;

        } else {
            cout << BRIGHT_RED
                 << "  Could not invert J^T J — covariance unavailable.\n"
                 << "  The solution may be at a boundary or the model"
                 << " may be ill-conditioned."
                 << RESET << endl;
        }
    }

    // ─────────────────────────────────────────────────────────────────
    //  Write iteration log
    // ─────────────────────────────────────────────────────────────────
    {
        string log_path = config.value("lm_log_path", "lm_iter_log.txt");
        ofstream logf(log_path);
        logf << "iter,sum_sq,chisq_lc,lambda,step_norm,gain_ratio,"
             << "fev,prior_scale,status\n";
        for (auto& e : iter_log) {
            logf << e.iter << "," << e.sum_sq << "," << e.chisq_lc
                 << "," << e.lambda << "," << e.step_norm
                 << "," << e.gain_ratio << "," << e.fev
                 << "," << e.prior_scale
                 << "," << e.status << "\n";
        }
        logf.close();
        cout << "Iteration log written to " << log_path << endl;
    }

    // ─────────────────────────────────────────────────────────────────
    //  Best-fit light curve, plot, output
    // ─────────────────────────────────────────────────────────────────
    model.set_param(best_pars);
    {
        vector<double> final_fit;
        double wd0, chisq0, wn0, lg10, lg20, rv10, rv20;
        Lcurve::light_curve_comp(model, data, scale, !no_file, false, sfac,
                                 final_fit, wd0, chisq0, wn0,
                                 lg10, lg20, rv10, rv20);
        if (device != "none" && device != "null")
            Helpers::plot_model(data, final_fit, no_file, copy, device);

        string sout = config["output_file_path"].get<string>();
        for (long unsigned int i = 0; i < data.size(); ++i)
            data[i].flux = final_fit[i] + noise * Subs::gauss2(seed);
        Helpers::write_data(data, sout);
    }

    // ─────────────────────────────────────────────────────────────────
    //  Persist results
    // ─────────────────────────────────────────────────────────────────
    {
        json lm_summary;
        lm_summary["converged"]        = converged;
        lm_summary["stop_reason"]      = stop_reason;
        lm_summary["iterations"]       = total_iter;
        lm_summary["function_evals"]   = fev_count;
        lm_summary["best_chisq_lc"]    = best_chisq;
        lm_summary["best_sum_sq"]      = best_sum_sq;
        lm_summary["final_lambda"]     = lambda;
        for (int i = 0; i < npar; ++i)
            lm_summary["best_pars"][names[i]] = (double)best_pars[i];
        config["lm_summary"] = lm_summary;
    }

    Helpers::write_config_and_model_to_json(
        model, config,
        config["output_file_path"].get<string>() + ".json");

    return 0;
}