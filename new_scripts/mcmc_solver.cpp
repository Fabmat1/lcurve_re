// mcmc_solver.cpp
// ═══════════════════════════════════════════════════════════════════════
//  Adaptive-Metropolis MCMC sampler for LCURVE binary-star light curves
//
//  Key features over the previous version
//  ───────────────────────────────────────
//  • Analytical physical prior  (replaces grid-based mass_ratio_pdf)
//  • Covariance reset after initial burn-in settling
//  • Scale-factor reset when switching to Cholesky proposals
//  • Bounded reflection with bounce cap
//  • Effective-sample-size (ESS) reporting
//  • Chain thinning for file output
//  • log-prior stored in chain for diagnostics
// ═══════════════════════════════════════════════════════════════════════

#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <deque>
#include <chrono>

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

// ═════════════════════════ terminal helpers ═══════════════════════════

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

std::size_t visual_length(const std::string& s)
{
    std::size_t len = 0;
    bool esc = false;
    for (char c : s) {
        if (esc) { if (c == 'm') esc = false; continue; }
        if (c == '\033') { esc = true; continue; }
        ++len;
    }
    return len;
}

// ═══════════════════ effective sample size (ESS) ═════════════════════
//
//  Uses Geyer's initial-positive-sequence (IPS) estimator:
//  sum consecutive pairs of autocorrelations, stop when a pair
//  sum first goes negative.  Conservative but robust.
//
double compute_ess(const vector<double>& x)
{
    const int n = static_cast<int>(x.size());
    if (n < 4) return static_cast<double>(n);

    double mean = 0.0;
    for (double xi : x) mean += xi;
    mean /= n;

    double c0 = 0.0;
    for (double xi : x) c0 += (xi - mean) * (xi - mean);
    c0 /= n;
    if (c0 < 1e-30) return static_cast<double>(n);

    double tau = 1.0;
    const int max_lag = min(n / 2, 10000);

    for (int lag = 1; lag < max_lag; lag += 2) {
        double rho1 = 0.0;
        for (int i = 0; i < n - lag; ++i)
            rho1 += (x[i] - mean) * (x[i + lag] - mean);
        rho1 /= (n * c0);

        double rho2 = 0.0;
        if (lag + 1 < n) {
            for (int i = 0; i < n - lag - 1; ++i)
                rho2 += (x[i] - mean) * (x[i + lag + 1] - mean);
            rho2 /= (n * c0);
        }

        if (rho1 + rho2 < 0.0) break;
        tau += 2.0 * (rho1 + rho2);
    }

    return n / max(tau, 1.0);
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
    json config = model_config.second;

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
    Subs::Array1D<double> ranges       = model.get_range();
    vector<pair<double,double>> limits  = model.get_limit();
    string device = config.value("plot_device", "none");

    // ── Gnuplot ──────────────────────────────────────────────────────
    Gnuplot gp;
    gp << "set terminal " + device + " title 'Live fitting plot'\n";
    gp << "set grid\n";

    // ── Identify parameter indices for the prior ─────────────────────
    int q_idx = -1, vs_idx = -1, r1_idx = -1, iangle_idx = -1;
    for (int i = 0; i < npar; ++i) {
        if      (names[i] == "q")              q_idx      = i;
        else if (names[i] == "velocity_scale") vs_idx     = i;
        else if (names[i] == "r1")             r1_idx     = i;
        else if (names[i] == "iangle")         iangle_idx = i;
    }

    cout << "Calculating MCMC for " << npar << " parameters:" << endl;
    for (int i = 0; i < npar; ++i)
        cout << "  " << names[i] << ": " << current_pars[i]
             << "  step " << dsteps[i]
             << "  limits [" << limits[i].first
             << ", " << limits[i].second << "]" << endl;

    // ─────────────────────────────────────────────────────────────────
    //  MCMC settings
    // ─────────────────────────────────────────────────────────────────
    int nsteps            = config.value("mcmc_steps",       25000);
    int burn_in           = config.value("mcmc_burn_in",     nsteps / 4);
    int thin              = max(1, config.value("mcmc_thin", 1));
    int progress_interval = config.value("progress_interval", 50);
    int max_model_points  = config.value("max_model_points",  500);

    // ANSI colours
    const string RESET        = "\033[0m";
    const string BRIGHT_GREEN = "\033[92m";
    const string BRIGHT_BLUE  = "\033[94m";
    const string BRIGHT_YELLOW= "\033[93m";
    const string BRIGHT_RED   = "\033[91m";
    const string BRIGHT_CYAN  = "\033[96m";
    const string BRIGHT_WHITE = "\033[97m";
    const string DIM          = "\033[2m";

    // ─────────────────────────────────────────────────────────────────
    //  Robbins-Monro step-size adaptation
    // ─────────────────────────────────────────────────────────────────
    bool   adapt_enabled       = config.value("adapt_enabled",        true);
    double adapt_target        = config.value("target_acceptance_rate",0.234);
    int    adapt_interval      = config.value("adapt_interval",       100);
    double adapt_rate          = config.value("adapt_rate",           1.0);
    double adapt_decay         = config.value("adapt_decay",          0.6);
    double adapt_min_stepscale = config.value("adapt_min_stepscale",  1e-4);
    double adapt_max_stepscale = config.value("adapt_max_stepscale",  1e4);
    if (npar == 1 && !config.contains("target_acceptance_rate"))
        adapt_target = 0.44;

    Subs::Array1D<double> initial_dsteps = dsteps;
    double adapt_log_scale      = 0.0;
    int    adapt_batch          = 0;
    int    adapt_window_accepts = 0;
    int    adapt_window_count   = 0;
    double adapt_current_rate   = -1.0;
    const double adapt_log_min  = std::log(adapt_min_stepscale);
    const double adapt_log_max  = std::log(adapt_max_stepscale);

    // ─────────────────────────────────────────────────────────────────
    //  Covariance adaptation  (Adaptive Metropolis, Haario+ 2001)
    //
    //  NEW: the Welford accumulator is RESET at `cov_reset_step`
    //  so that early burn-in transient does not corrupt the estimate.
    //  When the Cholesky first (re-)activates, the global scale
    //  factor is reset to 1 so the (2.38²/d)·Σ scaling is correct.
    // ─────────────────────────────────────────────────────────────────
    bool   adapt_covariance = config.value("adapt_covariance", true);
    int    cov_warmup       = config.value("cov_warmup",
                                  max(20 * npar, 500));
    double cov_epsilon      = config.value("cov_epsilon", 1e-6);
    double cov_sd           = 2.38 / std::sqrt(static_cast<double>(npar));

    // After reset, require fewer samples to re-activate Cholesky
    // (the chain should already be near the mode)
    int cov_min_after_reset = max(2 * npar + 2, 50);

    // Auto-compute reset step: 40 % of burn-in.  Set to 0 to disable.
    int cov_reset_step = config.value("cov_reset_step", -1);
    if (cov_reset_step < 0) cov_reset_step = burn_in * 2 / 5;

    // ── Covariance state ──
    int                    cov_n = 0;
    vector<double>         cov_mean(npar, 0.0);
    vector<vector<double>> cov_M2(npar, vector<double>(npar, 0.0));
    vector<vector<double>> chol_L(npar, vector<double>(npar, 0.0));
    bool                   cov_ready         = false;
    bool                   cov_has_been_reset = false;

    if (adapt_enabled || adapt_covariance)
    {
        cout << BRIGHT_CYAN << "Adaptation:" << RESET;
        if (adapt_enabled)
            cout << " Robbins-Monro (target "
                 << fixed << setprecision(1)
                 << (adapt_target * 100) << "%)";
        if (adapt_covariance) {
            cout << " + Covariance (warmup " << cov_warmup << " steps";
            if (cov_reset_step > 0)
                cout << ", reset at step " << cov_reset_step;
            cout << ")";
        }
        cout << endl;
    }

    // ── Cholesky decomposition of (2.38²/d)·Σ + ε·I ──
    auto try_update_cholesky = [&]() -> bool
    {
        if (cov_n < npar + 1) return false;
        const double inv_nm1 = 1.0 / (cov_n - 1);
        const double sd2     = cov_sd * cov_sd;

        vector<vector<double>> C(npar, vector<double>(npar));
        for (int i = 0; i < npar; ++i) {
            for (int j = 0; j < npar; ++j)
                C[i][j] = sd2 * cov_M2[i][j] * inv_nm1;
            C[i][i] += cov_epsilon;
        }

        vector<vector<double>> L(npar, vector<double>(npar, 0.0));
        for (int i = 0; i < npar; ++i) {
            for (int j = 0; j <= i; ++j) {
                double s = C[i][j];
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
        chol_L = std::move(L);
        return true;
    };

    // ── Reset helper ──
    auto reset_covariance = [&]()
    {
        cov_n = 0;
        fill(cov_mean.begin(), cov_mean.end(), 0.0);
        for (auto& row : cov_M2) fill(row.begin(), row.end(), 0.0);
        cov_ready = false;
        cov_has_been_reset = true;
    };

    // ─────────────────────────────────────────────────────────────────
    //  Physical priors  (replaces mass_ratio_pdf grid)
    // ─────────────────────────────────────────────────────────────────
    ObservedConstraints obs;
    bool use_priors = config.value("use_priors", false);
    double log_prior_current = 0.0;

    if (use_priors) {
        obs.P_days         = config.value("true_period", 1.0);
        obs.use_sin_i_prior = config.value("use_sin_i_prior", true);

        for (auto& [p, v] : config["priors"].items()) {
            auto [val, err_lo, err_hi] =
                Helpers::parseThreeDoubles(v.get<string>());
            if (err_hi <= 0.0) err_hi = err_lo;
            if (err_lo <= 0.0) err_lo = err_hi;

            if (p == "vrad1_obs") {
                obs.K_obs = val; obs.K_err_lo = err_lo;
                obs.K_err_hi = err_hi; obs.has_K = true;
            } else if (p == "m1") {
                obs.M1_obs = val; obs.M1_err_lo = err_lo;
                obs.M1_err_hi = err_hi; obs.has_M1 = true;
            } else if (p == "m2_min") {
                obs.M2min_obs = val; obs.M2min_err_lo = err_lo;
                obs.M2min_err_hi = err_hi; obs.has_M2min = true;
            } else if (p == "r1") {
                obs.R1_obs = val; obs.R1_err_lo = err_lo;
                obs.R1_err_hi = err_hi; obs.has_R1 = true;
            } else {
                cerr << "Unknown prior: " << p << endl;
                return 1;
            }
        }

        // ── Print active priors ──
        cout << BRIGHT_CYAN << "Physical priors (P = "
             << obs.P_days << " d):" << RESET << endl;
        if (obs.has_K)
            cout << "  K1     = " << obs.K_obs << " ± "
                 << obs.K_err_lo << "/" << obs.K_err_hi << " km/s\n";
        if (obs.has_M1)
            cout << "  M1     = " << obs.M1_obs << " ± "
                 << obs.M1_err_lo << "/" << obs.M1_err_hi << " M_sun\n";
        if (obs.has_M2min)
            cout << "  M2_min = " << obs.M2min_obs << " ± "
                 << obs.M2min_err_lo << "/" << obs.M2min_err_hi
                 << " M_sun (one-sided)\n";
        if (obs.has_R1)
            cout << "  R1     = " << obs.R1_obs << " ± "
                 << obs.R1_err_lo << "/" << obs.R1_err_hi << " R_sun\n";
        if (obs.use_sin_i_prior)
            cout << "  p(i) ~ sin(i)  (geometric prior)\n";

        // ── Warn about fixed parameters ──
        auto warn_fixed = [&](const string& pname, bool varied) {
            if (!varied)
                cout << BRIGHT_YELLOW << "  [WARNING] " << pname
                     << " is fixed — prior may be very restrictive"
                     << RESET << endl;
        };
        warn_fixed("q",              q_idx >= 0);
        warn_fixed("velocity_scale", vs_idx >= 0);
        warn_fixed("r1",             r1_idx >= 0);
        warn_fixed("iangle",         iangle_idx >= 0);

        // ── Evaluate & print initial prior ──
        double init_i  = (iangle_idx >= 0) ? current_pars[iangle_idx]
                                           : model.iangle.value;
        double init_q  = (q_idx >= 0)      ? current_pars[q_idx]
                                           : model.q.value;
        double init_vs = (vs_idx >= 0)     ? current_pars[vs_idx]
                                           : model.velocity_scale.value;
        double init_r1 = (r1_idx >= 0)     ? current_pars[r1_idx]
                                           : model.r1.value;

        log_prior_current = PhysicalPrior::compute(
                                init_i, init_q, init_vs, init_r1, obs);

        cout << BRIGHT_CYAN << "Initial implied quantities:" << RESET << endl;
        PhysicalPrior::print_implied(init_i, init_q, init_vs, init_r1, obs);
        cout << "  log-prior = " << log_prior_current << endl;

        if (log_prior_current < -50.0) {
            cout << BRIGHT_RED
                 << "  [WARNING] Initial parameters have very low prior"
                    " probability!\n"
                    "  Consider adjusting starting values to be more"
                    " consistent\n"
                    "  with the observational constraints."
                 << RESET << endl;
        }
    }

    // ─────────────────────────────────────────────────────────────────
    //  Sliding-window acceptance rate (post-burn-in)
    // ─────────────────────────────────────────────────────────────────
    int post_len    = nsteps - burn_in;
    int window_size = max(1, int(post_len * 0.1));
    deque<bool> acc_window;
    int window_accept_count = 0;

    // ─────────────────────────────────────────────────────────────────
    //  RNG
    // ─────────────────────────────────────────────────────────────────
    mt19937 rng(seed);
    normal_distribution<>        gauss(0.0, 1.0);
    uniform_real_distribution<>  uni(0.0, 1.0);

    // ─────────────────────────────────────────────────────────────────
    //  Initial χ²
    // ─────────────────────────────────────────────────────────────────
    vector<double> fit;
    double wd0, chisq0, wn0, lg10, lg20, rv10, rv20;
    Lcurve::light_curve_comp(model, data, scale, !no_file, false, sfac,
                             fit, wd0, chisq0, wn0,
                             lg10, lg20, rv10, rv20);

    double current_chisq = chisq0;
    double best_chisq    = chisq0;
    Subs::Array1D<double> best_pars = current_pars;
    vector<double> current_fit = fit;   // for live plotting

    // ─────────────────────────────────────────────────────────────────
    //  Chain storage
    // ─────────────────────────────────────────────────────────────────
    struct ChainEntry {
        int step;
        Subs::Array1D<double> pars;
        double chisq;
        double log_prior;
    };
    vector<ChainEntry> chain(nsteps - burn_in);

    // ─────────────────────────────────────────────────────────────────
    //  Adaptation bookkeeping: one call per step
    // ─────────────────────────────────────────────────────────────────
    auto record_step = [&](bool accepted_step)
    {
        // 1. Robbins-Monro global scale
        if (adapt_enabled) {
            ++adapt_window_count;
            if (accepted_step) ++adapt_window_accepts;

            if (adapt_window_count >= adapt_interval) {
                adapt_current_rate = double(adapt_window_accepts)
                                   / adapt_window_count;
                double gamma = adapt_rate
                             * std::pow(1.0 + adapt_batch, -adapt_decay);
                adapt_log_scale += gamma
                                 * (adapt_current_rate - adapt_target);
                adapt_log_scale  = std::clamp(adapt_log_scale,
                                              adapt_log_min, adapt_log_max);
                double sf = std::exp(adapt_log_scale);
                for (int i = 0; i < npar; ++i)
                    dsteps[i] = initial_dsteps[i] * sf;

                adapt_window_accepts = 0;
                adapt_window_count   = 0;
                ++adapt_batch;
            }
        }

        // 2. Welford online covariance
        if (adapt_covariance) {
            ++cov_n;
            vector<double> delta(npar);
            for (int i = 0; i < npar; ++i) {
                delta[i]     = current_pars[i] - cov_mean[i];
                cov_mean[i] += delta[i] / cov_n;
            }
            for (int i = 0; i < npar; ++i) {
                double d2i = current_pars[i] - cov_mean[i];
                for (int j = 0; j <= i; ++j) {
                    double d2j = current_pars[j] - cov_mean[j];
                    cov_M2[i][j] += delta[i] * d2j;
                    if (j < i) cov_M2[j][i] = cov_M2[i][j];
                }
            }
        }
    };

    // ─────────────────────────────────────────────────────────────────
    //  Timing
    // ─────────────────────────────────────────────────────────────────
    auto t_start = Clock::now();
    int  accepted = 0;

    // ═════════════════════════════════════════════════════════════════
    //  MCMC LOOP
    // ═════════════════════════════════════════════════════════════════
    for (int step = 0; step < nsteps; ++step) {

        // ── Covariance reset at configured step ──────────────────────
        if (adapt_covariance && cov_reset_step > 0
            && step == cov_reset_step && !cov_has_been_reset)
        {
            reset_covariance();
            cout << "\n" << BRIGHT_CYAN
                 << "  Covariance accumulator reset at step " << step
                 << " — rebuilding from clean samples"
                 << RESET << endl;
        }

        // ── Burn-in boundary info ────────────────────────────────────
        if (step == burn_in && (adapt_enabled || adapt_covariance))
        {
            double sf_now    = std::exp(adapt_log_scale);
            double gamma_now = adapt_rate
                             * std::pow(1.0 + adapt_batch, -adapt_decay);
            cout << "\n" << BRIGHT_CYAN
                 << "── Burn-in complete ─────────────────────────"
                 << RESET << endl;
            if (adapt_enabled) {
                cout << "  Target acceptance:  "
                     << fixed << setprecision(1)
                     << (adapt_target * 100) << "%" << endl;
                if (adapt_current_rate >= 0) {
                    double a = adapt_current_rate * 100;
                    bool ok = std::abs(a - adapt_target * 100) < 5;
                    cout << "  Current rate:       "
                         << (ok ? BRIGHT_GREEN : BRIGHT_YELLOW)
                         << fixed << setprecision(1) << a << "%"
                         << (ok ? " ok" : " ~") << RESET << endl;
                }
                cout << "  Scale factor:       x"
                     << fixed << setprecision(4) << sf_now << endl;
                cout << "  Current gamma:      "
                     << scientific << setprecision(2) << gamma_now
                     << "  (diminishing)" << RESET << endl;
            }
            if (adapt_covariance) {
                cout << "  Proposal mode:      "
                     << (cov_ready
                         ? BRIGHT_GREEN + string("Covariance (Cholesky)")
                         : BRIGHT_YELLOW + string("Diagonal"))
                     << RESET << endl;
            }
            cout << BRIGHT_CYAN
                 << "─────────────────────────────────────────────"
                 << RESET << endl;
        }

        // ── Progress display ─────────────────────────────────────────
        if (step % progress_interval == 0)
        {
            const double fraction = double(step) / nsteps;
            const int    percent  = int(fraction * 100.0 + 0.5);

            auto now       = Clock::now();
            double elapsed = chrono::duration<double>(now - t_start).count();
            double eta_s   = max(0.0,
                elapsed / max(fraction, 1e-8) - elapsed);
            int eta_i = int(eta_s);
            int h = eta_i/3600, m = (eta_i%3600)/60, s = eta_i%60;
            char eta_txt[16];
            (h)   ? snprintf(eta_txt, sizeof eta_txt,
                             "%02d:%02d:%02d", h, m, s)
                  : (m) ? snprintf(eta_txt, sizeof eta_txt,
                                   "%02d:%02d", m, s)
                        : snprintf(eta_txt, sizeof eta_txt,
                                   "%ds", s);

            double acc_rate = acc_window.empty() ? -1.0
                : 100.0 * window_accept_count / acc_window.size();

            // ── Build display line ───────────────────────────────────
            ostringstream oss_prefix;
            oss_prefix << "\r" << RESET << "[";

            struct Chunk { string txt; int prio; };
            vector<Chunk> chunks;

            // percentage
            { ostringstream t;
              if      (percent >= 90) t << BRIGHT_GREEN;
              else if (percent >= 50) t << BRIGHT_BLUE;
              else                    t << BRIGHT_YELLOW;
              t << setw(3) << percent << "%" << RESET;
              chunks.push_back({t.str(), 99});
            }

            // acceptance / adaptation
            { ostringstream t;
              if ((adapt_enabled||adapt_covariance) && step < burn_in) {
                  double dr = adapt_window_count > 0
                      ? 100.0*adapt_window_accepts/adapt_window_count
                      : (adapt_current_rate>=0 ? adapt_current_rate*100 : 0);
                  double dsf = std::exp(adapt_log_scale);
                  t << " | ";
                  if (abs(dr - adapt_target*100) < 5)       t << BRIGHT_GREEN;
                  else if (abs(dr - adapt_target*100) < 10) t << BRIGHT_YELLOW;
                  else                                      t << BRIGHT_RED;
                  t << "Adapt " << fixed << setprecision(1)
                    << dr << "->" << (adapt_target*100) << "%"
                    << RESET << DIM << " x" << setprecision(2) << dsf;
                  if (cov_ready) t << " Cov";
                  t << RESET;
              } else if (step < burn_in) {
                  t << DIM << " | Burn-in" << RESET;
              } else {
                  if (acc_rate < 0)
                      t << DIM << " | Acc --" << RESET;
                  else {
                      t << " | ";
                      if (acc_rate >= 20 && acc_rate <= 50)
                          t << BRIGHT_GREEN;
                      else if (acc_rate < 20) t << BRIGHT_RED;
                      else                    t << BRIGHT_YELLOW;
                      t << "Acc " << fixed << setprecision(1)
                        << acc_rate << "%" << RESET;
                      if (adapt_enabled)
                          t << DIM << " x" << setprecision(2)
                            << exp(adapt_log_scale) << RESET;
                      if (cov_ready) t << DIM << " Cov" << RESET;
                  }
              }
              chunks.push_back({t.str(), 30});
            }

            // step counter
            { ostringstream t;
              t << " | " << DIM << step << "/" << nsteps << RESET;
              chunks.push_back({t.str(), 20});
            }

            // ETA
            { ostringstream t;
              t << " | " << BRIGHT_CYAN << "ETA " << eta_txt << RESET;
              chunks.push_back({t.str(), 10});
            }

            sort(chunks.begin(), chunks.end(),
                 [](auto& a, auto& b){ return a.prio < b.prio; });

            const int cols    = tty_cols.load(memory_order_relaxed);
            const int min_bar = 6;

            for (size_t rm = 0; rm <= chunks.size(); ++rm) {
                string suffix;
                for (size_t i = rm; i < chunks.size(); ++i)
                    suffix += chunks[i].txt;
                suffix = "] " + suffix;

                int occupied = int(visual_length(oss_prefix.str())
                                 + visual_length(suffix));
                if (occupied + min_bar <= cols) {
                    int bar_cells  = cols - occupied;
                    int bar_filled = int(bar_cells * fraction);
                    string bar;
                    bar.reserve(bar_cells * 6);
                    for (int i = 0; i < bar_cells; ++i) {
                        if (i < bar_filled) {
                            if      (i < bar_cells*0.6)
                                bar += BRIGHT_GREEN + string("█") + RESET;
                            else if (i < bar_cells*0.8)
                                bar += BRIGHT_CYAN  + string("█") + RESET;
                            else
                                bar += BRIGHT_BLUE  + string("█") + RESET;
                        } else if (i == bar_filled && fraction < 1.0)
                            bar += BRIGHT_WHITE + string("▌") + RESET;
                        else
                            bar += DIM + string("░") + RESET;
                    }
                    cout << oss_prefix.str() << bar << suffix << flush;
                    break;
                }
            }
        }

        // ─────────────────────────────────────────────────────────────
        //  Propose new parameters
        // ─────────────────────────────────────────────────────────────
        Subs::Array1D<double> prop = current_pars;
        const double sf = std::exp(adapt_log_scale);

        if (cov_ready) {
            // Correlated proposal:  prop = current + sf · L · z
            vector<double> z(npar);
            for (int i = 0; i < npar; ++i) z[i] = gauss(rng);
            for (int i = 0; i < npar; ++i) {
                double offset = 0.0;
                for (int j = 0; j <= i; ++j)
                    offset += chol_L[i][j] * z[j];
                prop[i] = current_pars[i] + sf * offset;
            }
        } else {
            // Diagonal proposals with per-parameter steps
            for (int i = 0; i < npar; ++i)
                prop[i] = current_pars[i] + dsteps[i] * gauss(rng);
        }

        // ── Reflect off boundaries (with bounce cap) ────────────────
        for (int i = 0; i < npar; ++i) {
            double lo = limits[i].first, hi = limits[i].second;
            int bounces = 0;
            while ((prop[i] < lo || prop[i] > hi) && bounces < 20) {
                if (prop[i] < lo) prop[i] = 2*lo - prop[i];
                else              prop[i] = 2*hi - prop[i];
                ++bounces;
            }
            prop[i] = std::clamp(prop[i], lo, hi);  // safety fallback
        }
        model.set_param(prop);

        // ─────────────────────────────────────────────────────────────
        //  Evaluate prior (cheap — do BEFORE expensive model eval)
        // ─────────────────────────────────────────────────────────────
        double log_prior_prop = 0.0;
        if (use_priors) {
            double pi = (iangle_idx >= 0) ? prop[iangle_idx]
                                          : model.iangle.value;
            double pq = (q_idx >= 0)      ? prop[q_idx]
                                          : model.q.value;
            double pv = (vs_idx >= 0)     ? prop[vs_idx]
                                          : model.velocity_scale.value;
            double pr = (r1_idx >= 0)     ? prop[r1_idx]
                                          : model.r1.value;

            log_prior_prop = PhysicalPrior::compute(pi, pq, pv, pr, obs);

            // Skip expensive model evaluation for hopeless proposals
            if (log_prior_prop <= -1e29) {
                model.set_param(current_pars);
                record_step(false);
                if (step >= burn_in)
                    chain[step - burn_in] = {step - burn_in,
                                             current_pars,
                                             current_chisq,
                                             log_prior_current};
                continue;
            }
        }

        // ─────────────────────────────────────────────────────────────
        //  Evaluate proposed model
        // ─────────────────────────────────────────────────────────────
        vector<double> fitp;
        double wdp, chp, wnp, lg1p, lg2p, rv1p, rv2p;

        try {
            light_curve_comp_fast(model, data, scale, !no_file, false, sfac,
                                 fitp, wdp, chp, wnp,
                                 lg1p, lg2p, rv1p, rv2p,
                                 max_model_points);
        }
        catch (Lcurve::Lcurve_Error&) {
            model.set_param(current_pars);
            record_step(false);
            if (step >= burn_in)
                chain[step - burn_in] = {step - burn_in,
                                         current_pars,
                                         current_chisq,
                                         log_prior_current};
            continue;
        }

        // ─────────────────────────────────────────────────────────────
        //  Accept / reject
        // ─────────────────────────────────────────────────────────────
        bool this_accept = false;
        double log_alpha = -0.5 * (chp - current_chisq)
                         + (log_prior_prop - log_prior_current);

        if (log_alpha >= 0.0 || log(uni(rng)) < log_alpha) {
            current_pars      = prop;
            log_prior_current = log_prior_prop;
            current_chisq     = chp;
            current_fit       = fitp;
            ++accepted;
            this_accept = true;
            if (chp < best_chisq) {
                best_chisq = chp;
                best_pars  = prop;
            }
        } else {
            model.set_param(current_pars);
        }

        // ── Feed adaptation ──────────────────────────────────────────
        record_step(this_accept);

        // ── Cholesky update ──────────────────────────────────────────
        if (adapt_covariance)
        {
            int threshold = cov_has_been_reset ? cov_min_after_reset
                                               : cov_warmup;
            if (cov_n >= threshold
                && (!cov_ready || step % adapt_interval == 0))
            {
                bool ok = try_update_cholesky();
                if (ok && !cov_ready) {
                    cov_ready = true;
                    // ── Reset scale so (2.38²/d)·Σ is the starting
                    //    proposal — Robbins-Monro fine-tunes from here
                    adapt_log_scale = 0.0;
                    adapt_batch     = 0;
                    adapt_window_accepts = 0;
                    adapt_window_count   = 0;
                    for (int i = 0; i < npar; ++i)
                        dsteps[i] = initial_dsteps[i];
                    cout << "\n" << BRIGHT_CYAN
                         << "  Cholesky proposal activated at step "
                         << step << " (cov_n = " << cov_n << ")"
                         << RESET << endl;
                }
                if (!ok && cov_ready) {
                    cov_ready = false;   // fall back to diagonal
                }
            }
        }

        // ── Post-burn-in bookkeeping ─────────────────────────────────
        if (step >= burn_in) {
            acc_window.push_back(this_accept);
            if (this_accept) ++window_accept_count;
            if (int(acc_window.size()) > window_size) {
                if (acc_window.front()) --window_accept_count;
                acc_window.pop_front();
            }
            chain[step - burn_in] = {step - burn_in,
                                     current_pars,
                                     current_chisq,
                                     log_prior_current};
        }

        // ── Live plot (current accepted fit) ─────────────────────────
        if (step % progress_interval == 0)
            Helpers::plot_model_live(data, current_fit, no_file, copy, gp);

    } // ═══ end MCMC loop ═══

    // ─────────────────────────────────────────────────────────────────
    //  Timing
    // ─────────────────────────────────────────────────────────────────
    auto t_end   = Clock::now();
    auto dur_ms  = chrono::duration_cast<chrono::milliseconds>(
                       t_end - t_start).count();
    double total = dur_ms / 1000.0;
    int    hrs   = int(total / 3600);
    int    mins  = int((total - hrs*3600) / 60);
    double secs  = total - hrs*3600 - mins*60;

    cout << "\n" << BRIGHT_GREEN << "MCMC sampling completed! Took: ";
    if (hrs  > 0) cout << hrs  << "h ";
    if (mins > 0) cout << mins << "m ";
    cout << fixed << setprecision(2) << secs << "s" << RESET << endl;

    // ─────────────────────────────────────────────────────────────────
    //  Adaptation report
    // ─────────────────────────────────────────────────────────────────
    if (adapt_enabled) {
        double final_sf = std::exp(adapt_log_scale);
        double final_g  = adapt_rate
                        * std::pow(1.0 + adapt_batch, -adapt_decay);
        cout << BRIGHT_CYAN << "Final adaptation state:" << RESET << endl;
        cout << "  Total batches:    " << adapt_batch << endl;
        cout << "  Final gamma:      " << scientific << setprecision(2)
             << final_g << endl;
        cout << "  Final scale:      x" << fixed << setprecision(4)
             << final_sf << endl;
        cout << "  Last batch rate:  " << setprecision(1)
             << (adapt_current_rate * 100.0) << "%"
             << " (target " << (adapt_target * 100.0) << "%)" << endl;
    }

    // ─────────────────────────────────────────────────────────────────
    //  Effective Sample Size
    // ─────────────────────────────────────────────────────────────────
    {
        cout << BRIGHT_CYAN << "\nEffective sample sizes:" << RESET << endl;
        size_t mw = 0;
        for (int i = 0; i < npar; ++i)
            mw = max(mw, names[i].size());

        double min_ess = 1e30;
        for (int p = 0; p < npar; ++p) {
            vector<double> trace(chain.size());
            for (size_t s = 0; s < chain.size(); ++s)
                trace[s] = chain[s].pars[p];
            double ess = compute_ess(trace);
            min_ess = min(min_ess, ess);

            string col = BRIGHT_GREEN;
            if      (ess < 100)  col = BRIGHT_RED;
            else if (ess < 500)  col = BRIGHT_YELLOW;

            cout << "  " << setw(int(mw)) << left << names[p]
                 << "  " << col << fixed << setprecision(0)
                 << setw(8) << right << ess << RESET
                 << " / " << chain.size() << endl;
        }

        // ESS for chi-squared
        {
            vector<double> ctrace(chain.size());
            for (size_t s = 0; s < chain.size(); ++s)
                ctrace[s] = chain[s].chisq;
            double cess = compute_ess(ctrace);
            cout << "  " << setw(int(mw)) << left << "chisq"
                 << "  " << fixed << setprecision(0)
                 << setw(8) << right << cess
                 << " / " << chain.size() << endl;
        }

        if (min_ess < 100) {
            cout << BRIGHT_RED
                 << "  [WARNING] Some ESS < 100 — chain may not have"
                    " converged.\n"
                    "  Consider running longer or improving step sizes."
                 << RESET << endl;
        }
    }

    // ─────────────────────────────────────────────────────────────────
    //  Correlation matrix
    // ─────────────────────────────────────────────────────────────────
    if (adapt_covariance && cov_n > 1)
    {
        const double inv_nm1 = 1.0 / (cov_n - 1);
        vector<double> sd(npar);
        for (int i = 0; i < npar; ++i)
            sd[i] = std::sqrt(cov_M2[i][i] * inv_nm1);

        if (npar <= 20) {
            cout << BRIGHT_CYAN << "\nParameter correlations:"
                 << RESET << endl;
            size_t mw = 0;
            for (int i = 0; i < npar; ++i)
                mw = max(mw, names[i].size());
            const int cw = max(8, int(mw) + 1);

            cout << "  " << setw(int(mw)) << " ";
            for (int j = 0; j < npar; ++j)
                cout << setw(cw) << right
                     << names[j].substr(0, cw - 1);
            cout << endl;

            for (int i = 0; i < npar; ++i) {
                cout << "  " << setw(int(mw)) << left << names[i];
                for (int j = 0; j < npar; ++j) {
                    double r = (sd[i]>0 && sd[j]>0)
                        ? cov_M2[i][j]*inv_nm1 / (sd[i]*sd[j])
                        : 0.0;
                    string col = RESET;
                    if (i == j)              col = DIM;
                    else if (abs(r) > .7)    col = BRIGHT_RED;
                    else if (abs(r) > .4)    col = BRIGHT_YELLOW;
                    cout << col << setw(cw) << right
                         << fixed << setprecision(2) << r << RESET;
                }
                cout << endl;
            }
        }

        // Effective step sizes
        double sf_final = std::exp(adapt_log_scale);
        cout << BRIGHT_CYAN
             << "\nEffective step sizes (paste into model file):"
             << RESET << endl;
        for (int i = 0; i < npar; ++i)
            cout << "  " << names[i] << "  "
                 << scientific << setprecision(6)
                 << (sf_final * cov_sd * sd[i]) << endl;

        // Save to JSON
        json cov_info;
        for (int i = 0; i < npar; ++i) {
            cov_info["marginal_sigma"][names[i]] = sd[i];
            cov_info["effective_stepsizes"][names[i]] =
                sf_final * cov_sd * sd[i];
            for (int j = 0; j < npar; ++j) {
                double r = (sd[i]>0 && sd[j]>0)
                    ? cov_M2[i][j]*inv_nm1 / (sd[i]*sd[j]) : 0.0;
                string key = names[i] + "," + names[j];
                cov_info["correlations"][key]  = r;
                cov_info["covariance"][key]    = cov_M2[i][j]*inv_nm1;
            }
        }
        config["adaptation_results"]["covariance"] = cov_info;
    }

    // ─────────────────────────────────────────────────────────────────
    //  Best-fit report
    // ─────────────────────────────────────────────────────────────────
    cout << "\nBest chi^2 = " << best_chisq << endl;
    for (int i = 0; i < npar; ++i)
        cout << "  " << names[i] << " = " << best_pars[i] << endl;

    if (use_priors) {
        double bi = (iangle_idx>=0) ? best_pars[iangle_idx]
                                    : model.iangle.value;
        double bq = (q_idx>=0)      ? best_pars[q_idx]
                                    : model.q.value;
        double bv = (vs_idx>=0)     ? best_pars[vs_idx]
                                    : model.velocity_scale.value;
        double br = (r1_idx>=0)     ? best_pars[r1_idx]
                                    : model.r1.value;
        cout << BRIGHT_CYAN << "Best-fit implied quantities:"
             << RESET << endl;
        PhysicalPrior::print_implied(bi, bq, bv, br, obs);
    }

    // ─────────────────────────────────────────────────────────────────
    //  Write chain (with thinning)
    // ─────────────────────────────────────────────────────────────────
    {
        string chain_path = config.value("chain_out_path", "chain_out.txt");
        ofstream chain_file(chain_path);
        chain_file << "step";
        for (auto& n : names) chain_file << "," << n;
        chain_file << ",chisq,log_prior\n";

        int written = 0;
        for (int i = 0; i < int(chain.size()); i += thin) {
            auto& e = chain[i];
            chain_file << e.step;
            for (int j = 0; j < npar; ++j)
                chain_file << "," << e.pars[j];
            chain_file << "," << e.chisq
                       << "," << e.log_prior << "\n";
            ++written;
        }
        chain_file.close();
        cout << "Chain written to " << chain_path
             << " (" << written << " rows, thin=" << thin << ")" << endl;
    }

    // ─────────────────────────────────────────────────────────────────
    //  Best-fit light curve, plot, output
    // ─────────────────────────────────────────────────────────────────
    vector<double> best_fit;
    model.set_param(best_pars);
    Lcurve::light_curve_comp(model, data, scale, !no_file, false, sfac,
                             best_fit, wd0, chisq0, wn0,
                             lg10, lg20, rv10, rv20);
    if (device != "none" && device != "null")
        Helpers::plot_model(data, best_fit, no_file, copy, device);

    string sout = config["output_file_path"].get<string>();
    for (long unsigned int i = 0; i < data.size(); ++i)
        data[i].flux = best_fit[i] + noise * Subs::gauss2(seed);
    Helpers::write_data(data, sout);

    // ─────────────────────────────────────────────────────────────────
    //  Persist adaptation state
    // ─────────────────────────────────────────────────────────────────
    if (adapt_enabled) {
        json adapt_out;
        adapt_out["final_scale_factor"]        = std::exp(adapt_log_scale);
        adapt_out["last_batch_acceptance_rate"] = adapt_current_rate;
        adapt_out["target_acceptance_rate"]     = adapt_target;
        adapt_out["total_batches"]             = adapt_batch;
        adapt_out["final_learning_rate"]       =
            adapt_rate * std::pow(1.0 + adapt_batch, -adapt_decay);
        for (int i = 0; i < npar; ++i) {
            adapt_out["initial_stepsizes"][names[i]] = initial_dsteps[i];
            adapt_out["adapted_stepsizes"][names[i]] = dsteps[i];
        }
        config["adaptation_results"] = adapt_out;
    }

    Helpers::write_config_and_model_to_json(
        model, config,
        config["output_file_path"].get<string>() + ".json");

    return 0;
}