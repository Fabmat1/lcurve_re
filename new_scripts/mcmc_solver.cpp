#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <deque>
#include <chrono>
#include <nlohmann/json.hpp>
#include "../src/lcurve_base/lcurve.h"
#include "../src/new_helpers.h"
#include "../src/new_subs.h"
#include "../src/mass_ratio_pdf.h"
#include "../src/grid_cache.h"
#include <cmath>
#include <sys/ioctl.h>   // TIOCGWINSZ
#include <unistd.h>      // STDOUT_FILENO
#include <signal.h>
#include <atomic>

using namespace std;
using json = nlohmann::json;
using Clock = chrono::steady_clock;

inline int current_tty_columns()
{
    winsize ws{};
    if (::isatty(STDOUT_FILENO) == 0) {
        return 80;
    }

    if (::ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == -1) {
        return 80;
    }

    return ws.ws_col ? ws.ws_col : 80;
}

std::atomic<int> tty_cols{ current_tty_columns() };

void sigwinch_handler(int) {
    tty_cols.store(current_tty_columns(), std::memory_order_relaxed);
}


// Add these function definitions (or in a header file)
double log_gaussian_pdf(double x, double mean, double sigma) {
    double z = (x - mean) / sigma;
    return -0.5 * z * z - 0.5 * log(2.0 * M_PI) - log(sigma);
}

// For asymmetric errors, use a split normal distribution
double log_split_normal_pdf(double x, double mean, double sigma_left, double sigma_right) {
    if (x < mean) {
        return log_gaussian_pdf(x, mean, sigma_left);
    } else {
        return log_gaussian_pdf(x, mean, sigma_right);
    }
}

std::size_t visual_length(const std::string& s)
{
    std::size_t len = 0;
    bool esc = false;
    for (char c : s) {
        if (esc) {                    // inside "\033[ …"
            if (c == 'm') esc = false;
            continue;
        }
        if (c == '\033') { esc = true; continue; }
        ++len;
    }
    return len;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <config_file.json>" << endl;
        return 1;
    }
    ::signal(SIGWINCH, sigwinch_handler);    // react to window-resize
    // Load model and configuration
    string config_file = argv[1];
    auto model_config = Helpers::load_model_and_config_from_json(config_file);
    Lcurve::Model model = model_config.first;
    json config = model_config.second;

    // Load data (or fake data)
    auto data_copy = Helpers::read_and_copy_lightcurve_from_file(config["data_file_path"]);
    Lcurve::Data data = data_copy.first;
    Lcurve::Data copy = data_copy.second;
    bool no_file = data.empty();
    double noise = 0.0;
    if (no_file) throw Lcurve::Lcurve_Error("No data file provided");
    else noise = config["noise"].get<double>();

    // Initialize scale factors
    int seed;
    bool scale;
    vector<double> sfac;
    Helpers::load_seed_scale_sfac(config, no_file, model, seed, scale, sfac);

    // Get variable parameters
    int npar = model.nvary();
    vector<string> names(npar);
    for (int i = 0; i < npar; ++i) names[i] = model.get_name(i);
    Subs::Array1D<double> current_pars = model.get_param();
    Subs::Array1D<double> dsteps      = model.get_dstep();
    Subs::Array1D<double> ranges      = model.get_range();
    vector<pair<double, double>> limits = model.get_limit();
    string device = config.value("plot_device", "none");

    // Define Gnuplot instance
    Gnuplot gp; // Open once, reuse
    gp << "set terminal " + device + " title 'Live fitting plot'\n";
    gp << "set grid\n";

    cout << "Calculating MCMC for " << npar << " parameters:" << endl;
    for (int i = 0; i < npar; ++i) {
        cout << names[i] << ": " << current_pars[i]
             << " with stepsize " << dsteps[i]
             << " and limits from " << limits[i].first
             << " to " << limits[i].second << endl;
    }

    // MCMC settings
    int nsteps            = config.value("mcmc_steps", 25000);
    int burn_in           = config.value("mcmc_burn_in", nsteps/4);
    int progress_interval = config.value("progress_interval", 50);
    int max_model_points  = config.value("max_model_points", 500);

    // ANSI color codes
    const string RESET = "\033[0m";
    const string BRIGHT_GREEN = "\033[92m";
    const string BRIGHT_BLUE = "\033[94m";
    const string BRIGHT_YELLOW = "\033[93m";
    const string BRIGHT_RED = "\033[91m";
    const string BRIGHT_CYAN = "\033[96m";
    const string BRIGHT_WHITE = "\033[97m";
    const string DIM = "\033[2m";
    
    // ═══════════════════════════════════════════════════════════════════
    //  Adaptive step-size configuration  (all tuneable from JSON config)
    //
    //  Config keys:
    //    adapt_enabled          – bool   (true)   master on/off switch
    //    target_acceptance_rate – double (0.234)   optimal for d≥2; auto 0.44 for d=1
    //    adapt_interval         – int    (100)     steps per adaptation batch
    //    adapt_rate             – double (1.0)     Robbins-Monro learning rate c₀
    //    adapt_decay            – double (0.6)     power-law exponent β for γ decay
    //    adapt_min_stepscale    – double (1e-4)    floor on global scale factor
    //    adapt_max_stepscale    – double (1e4)     ceiling on global scale factor
    //
    //  The learning rate γ_n = c₀ / (1+n)^β decays to zero, satisfying
    //  the Diminishing Adaptation condition (Roberts & Rosenthal 2007)
    //  so the chain converges to the correct target distribution even
    //  with continuous adaptation throughout the entire run.
    // ═══════════════════════════════════════════════════════════════════
    bool   adapt_enabled       = config.value("adapt_enabled", true);
    double adapt_target        = config.value("target_acceptance_rate", 0.234);
    int    adapt_interval      = config.value("adapt_interval", 100);
    double adapt_rate          = config.value("adapt_rate", 1.0);
    double adapt_decay         = config.value("adapt_decay", 0.6);
    double adapt_min_stepscale = config.value("adapt_min_stepscale", 1e-4);
    double adapt_max_stepscale = config.value("adapt_max_stepscale", 1e4);

    // For 1-D problems the theoretical optimum is ≈ 0.44, not 0.234
    if (npar == 1 && !config.contains("target_acceptance_rate"))
        adapt_target = 0.44;

    // ── Adaptation state ──
    Subs::Array1D<double> initial_dsteps = dsteps;   // pristine copy
    double adapt_log_scale        = 0.0;
    int    adapt_batch            = 0;
    int    adapt_window_accepts   = 0;
    int    adapt_window_count     = 0;
    double adapt_current_rate     = -1.0;
    const double adapt_log_min    = std::log(adapt_min_stepscale);
    const double adapt_log_max    = std::log(adapt_max_stepscale);

    if (adapt_enabled) {
        cout << BRIGHT_CYAN << "Step-size adaptation:" << RESET
             << " target " << fixed << setprecision(1) << (adapt_target * 100)
             << "%, interval " << adapt_interval
             << ", continuous (Robbins-Monro, decay β="
             << setprecision(2) << adapt_decay << ")" << endl;
    }

    // Determine if we should use grid caching
    long total_evaluations = nsteps;
    bool use_grid_cache = (total_evaluations > 100000) && (npar <= 6);

    double grid_threshold = 0.5;

    std::unique_ptr<GridCache> grid_cache;
    if (use_grid_cache) {
        cout << "Using optimized grid cache for " << npar << " parameters" << endl;
        vector<double> steps_vec(ranges.begin(), ranges.end());
        vector<double> initial_vec(current_pars.begin(), current_pars.end());
        grid_cache = std::make_unique<GridCache>(npar, initial_vec, steps_vec, limits, 50000);
    }


    // prepare sliding-window acceptance-rate over last 10% of post-burn-in
    int post_len = nsteps - burn_in;
    int window_size = max(1, int(post_len * 0.1));
    deque<bool> acc_window;
    int window_accept_count = 0;

    // RNG
    mt19937 rng(seed);
    normal_distribution<> gauss(0.0, 1.0);
    uniform_real_distribution<> uni(0.0, 1.0);

    // Initial chi-squared
    vector<double> fit;
    double wd0, chisq0, wn0, lg10, lg20, rv10, rv20;
    Lcurve::light_curve_comp(model, data, scale, !no_file, false, sfac,
                             fit, wd0, chisq0, wn0,
                             lg10, lg20, rv10, rv20);

    double current_chisq = chisq0;
    double best_chisq    = chisq0;
    Subs::Array1D<double> best_pars = current_pars;

    // Prepare in-memory chain storage
    struct ChainEntry { int step; Subs::Array1D<double> pars; double chisq; };
    vector<ChainEntry> chain(nsteps - burn_in);

    tuple<double, double, double> vobs_prior;
    tuple<double, double, double> m1_prior;
    tuple<double, double, double> m2_min_prior;
    tuple<double, double, double> r1_prior;
    bool prior_warning_printed = false;

    int q_idx = -1;
    int vs_idx = -1;
    int r1_idx = -1;
    int iangle_idx = -1;

    for (int i = 0; i < npar; ++i) {
        if (names[i] == "q") q_idx = i;
        else if (names[i] == "velocity_scale") vs_idx = i;
        else if (names[i] == "r1") r1_idx = i;
        else if (names[i] == "iangle") iangle_idx = i;
    }

    double log_prior_current = 0.0;
    bool use_priors = config.value("use_priors", false);
    if (use_priors) {
        for (auto &[p, v] : config["priors"].items()) {
            tuple<double, double, double> val_err_arr = Helpers::parseThreeDoubles(v.get<string>());
            if (p == "vrad1_obs") {
                vobs_prior = val_err_arr;
            }
            else if (p == "m1") {
                m1_prior = val_err_arr;
            }
            else if (p == "m2_min") {
                m2_min_prior = val_err_arr;
            }
            else if (p == "r1") {
                r1_prior = val_err_arr;
            }
            else {
                cerr << "Unknown prior: " << p << endl;
                return 1;
            }
        }

        double m1_mean = get<0>(m1_prior);
        double m1_err = get<1>(m1_prior);
        double m2_mean = get<0>(m2_min_prior);
        double m2_err = get<1>(m2_min_prior);
        double K_mean = get<0>(vobs_prior);
        double K_err = get<1>(vobs_prior);
        double R_mean = get<0>(r1_prior);
        double R_err = get<1>(r1_prior);
        initialize_mass_ratio_pdf_grid(m1_mean, m1_err, m2_mean, m2_err,
                                       K_mean, K_err, R_mean, R_err,
                                       config.value("true_period", 1.0), 0.0000001);

        double init_incl = model.iangle.vary && iangle_idx >= 0 ? current_pars[iangle_idx] : model.iangle.value;
        double init_q = model.q.vary && q_idx >= 0 ? current_pars[q_idx] : model.q.value;
        double init_vs = model.velocity_scale.vary && vs_idx >= 0 ? current_pars[vs_idx] : model.velocity_scale.value;
        double init_r1 = model.r1.vary && r1_idx >= 0 ? current_pars[r1_idx] : model.r1.value;

        log_prior_current += log_mass_ratio_pdf(init_incl, init_q, init_vs, init_r1);
    }

    // Timing
    auto t_start = Clock::now();
    int accepted = 0;

    // ── Declare loop counter outside for lambda capture ──
    int step = 0;

    // ── Lambda: record one proposal outcome, fire adaptation when batch full ──
    //    Called every step, throughout the entire run.  The decaying learning
    //    rate γ_n = c₀/(1+n)^β ensures late-stage updates are negligible.
    auto record_adapt = [&](bool accepted_step) {
        if (!adapt_enabled) return;
        ++adapt_window_count;
        if (accepted_step) ++adapt_window_accepts;

        if (adapt_window_count >= adapt_interval) {
            adapt_current_rate = double(adapt_window_accepts) / adapt_window_count;

            // Robbins–Monro update with power-law decaying learning rate
            double gamma = adapt_rate * std::pow(1.0 + adapt_batch, -adapt_decay);
            adapt_log_scale += gamma * (adapt_current_rate - adapt_target);
            adapt_log_scale  = std::clamp(adapt_log_scale, adapt_log_min, adapt_log_max);

            double sf = std::exp(adapt_log_scale);
            for (int i = 0; i < npar; ++i)
                dsteps[i] = initial_dsteps[i] * sf;

            adapt_window_accepts = 0;
            adapt_window_count   = 0;
            ++adapt_batch;
        }
    };

    // MCMC loop
    for (step = 0; step < nsteps; ++step) {

        // ── Snapshot at burn-in boundary (informational, adaptation continues) ──
        if (adapt_enabled && step == burn_in) {
            double sf_now = std::exp(adapt_log_scale);
            double gamma_now = adapt_rate * std::pow(1.0 + adapt_batch, -adapt_decay);
            cout << "\n" << BRIGHT_CYAN
                 << "── Burn-in complete · adaptation snapshot ────────"
                 << RESET << endl;
            cout << "  Target acceptance:  "
                 << fixed << setprecision(1) << (adapt_target * 100.0) << "%" << endl;
            if (adapt_current_rate >= 0.0) {
                double achieved = adapt_current_rate * 100.0;
                bool on_target = std::abs(achieved - adapt_target * 100.0) < 5.0;
                cout << "  Current rate:       "
                     << (on_target ? BRIGHT_GREEN : BRIGHT_YELLOW)
                     << fixed << setprecision(1) << achieved << "%"
                     << (on_target ? " ✓" : " ~") << RESET << endl;
            }
            cout << "  Batches so far:     " << adapt_batch << endl;
            cout << "  Current γ:          " << scientific << setprecision(2)
                 << gamma_now << "  (diminishing)" << RESET << endl;
            cout << "  Scale factor:       ×" << fixed << setprecision(4)
                 << sf_now << endl;
            size_t max_name = 0;
            for (int i = 0; i < npar; ++i)
                max_name = std::max(max_name, names[i].size());
            cout << "  Step sizes:" << endl;
            for (int i = 0; i < npar; ++i) {
                cout << "    " << setw(static_cast<int>(max_name)) << left << names[i]
                     << "  " << scientific << setprecision(4) << initial_dsteps[i]
                     << " → " << dsteps[i] << endl;
            }
            cout << BRIGHT_CYAN
                 << "  Adaptation continues with diminishing γ"
                 << RESET << endl;
            cout << BRIGHT_CYAN
                 << "──────────────────────────────────────────────────"
                 << RESET << endl;
        }

        /*------ adaptive, never-overflowing, auto-shrinking progress line --------*/
        if (step % progress_interval == 0)
        {

            /* ---------- fraction, percentage and ETA -------------------------- */
            const double fraction  = double(step) / nsteps;
            const int    percent   = int(fraction * 100.0 + 0.5);

            auto now       = Clock::now();
            double elapsed = chrono::duration<double>(now - t_start).count();
            double eta_s   = elapsed / std::max(fraction, 1e-8) - elapsed;
            eta_s          = std::max(0.0, eta_s);
            int eta_i      = int(eta_s);
            int h = eta_i / 3600, m = (eta_i % 3600) / 60, s = eta_i % 60;
            char eta_txt[16];
            (h)   ? snprintf(eta_txt, sizeof eta_txt, "%02d:%02d:%02d", h, m, s)
                  : (m) ? snprintf(eta_txt, sizeof eta_txt, "%02d:%02d", m, s)
                         : snprintf(eta_txt, sizeof eta_txt, "%ds", s);

            /* ---------- acceptance rate (post-burn-in window) ----------------- */
            double acc_rate = acc_window.empty() ? -1.0
                                                 : 100.0 * window_accept_count / acc_window.size();

            /* ---------- build the prefix ("MCMC [") --------------------------- */
            std::ostringstream oss_prefix;
            oss_prefix << "\r" << RESET << "[";

            /* ---------- build the variable suffix parts ----------------------- */
            struct Chunk { std::string txt; int prio; };
            std::vector<Chunk> chunks;

            /* percentage ------------------------------------------------------- */
            {
                std::ostringstream tmp;
                if (percent >= 90)        tmp << BRIGHT_GREEN;
                else if (percent >= 50)   tmp << BRIGHT_BLUE;
                else                      tmp << BRIGHT_YELLOW;
                tmp << std::setw(3) << percent << "%" << RESET;
                chunks.push_back({tmp.str(), 99});
            }

            /* acceptance rate / adaptation info -------------------------------- */
            {
                std::ostringstream tmp;

                if (adapt_enabled && step < burn_in) {
                    // ── During burn-in: show live batch rate → target, ×scale ──
                    double disp_rate = adapt_window_count > 0
                        ? 100.0 * adapt_window_accepts / adapt_window_count
                        : (adapt_current_rate >= 0.0 ? adapt_current_rate * 100.0 : 0.0);
                    double disp_sf = std::exp(adapt_log_scale);
                    tmp << " │ ";
                    if (std::abs(disp_rate - adapt_target * 100.0) < 5.0)
                        tmp << BRIGHT_GREEN;
                    else if (std::abs(disp_rate - adapt_target * 100.0) < 10.0)
                        tmp << BRIGHT_YELLOW;
                    else
                        tmp << BRIGHT_RED;
                    tmp << "Adapt " << std::fixed << std::setprecision(1)
                        << disp_rate << "→" << (adapt_target * 100.0) << "%"
                        << RESET << DIM << " ×" << std::setprecision(2)
                        << disp_sf << RESET;
                }
                else if (!adapt_enabled && step < burn_in) {
                    tmp << DIM << " │ Burn-in" << RESET;
                }
                else {
                    // ── Post-burn-in: sliding window acceptance rate ──
                    if (acc_rate < 0)
                        tmp << DIM << " │ Acc --" << RESET;
                    else {
                        tmp << " │ ";
                        if (acc_rate >= 20 && acc_rate <= 50)
                            tmp << BRIGHT_GREEN;
                        else if (acc_rate < 20)
                            tmp << BRIGHT_RED;
                        else
                            tmp << BRIGHT_YELLOW;
                        tmp << "Acc " << std::fixed << std::setprecision(1)
                            << acc_rate << "%" << RESET;
                        // Dim scale-factor reminder that adaptation is still active
                        if (adapt_enabled)
                            tmp << DIM << " ×" << std::setprecision(2)
                                << std::exp(adapt_log_scale) << RESET;
                    }
                }
                chunks.push_back({tmp.str(), 30});
            }

            /* step counter ----------------------------------------------------- */
            {
                std::ostringstream tmp;
                tmp << " │ " << DIM << step << "/" << nsteps << RESET;
                chunks.push_back({tmp.str(), 20});
            }

            /* ETA -------------------------------------------------------------- */
            {
                std::ostringstream tmp;
                tmp << " │ " << BRIGHT_CYAN << "⏱ " << eta_txt << RESET;
                chunks.push_back({tmp.str(), 10});
            }

            /* sort by priority (lowest prio removed first) --------------------- */
            std::sort(chunks.begin(), chunks.end(),
                      [](const Chunk& a, const Chunk& b){ return a.prio < b.prio; });

            /* ---------- decide what can still be shown ------------------------ */
            const int cols = tty_cols.load(std::memory_order_relaxed);
            const int min_bar = 6;

            for (std::size_t remove = 0; remove <= chunks.size(); ++remove)
            {
                std::string suffix;
                for (std::size_t i = remove; i < chunks.size(); ++i)
                    suffix += chunks[i].txt;
                suffix = "] " + suffix;

                const int occupied =
                    int(visual_length(oss_prefix.str()) + visual_length(suffix));

                if (occupied + min_bar <= cols)
                {
                    const int bar_cells   = cols - occupied;
                    const int bar_filled  = int(bar_cells * fraction);

                    std::string bar; bar.reserve(bar_cells);
                    for (int i = 0; i < bar_cells; ++i)
                    {
                        if (i < bar_filled) {
                            if      (i < bar_cells * 0.6)
                                bar += BRIGHT_GREEN + std::string("█") + RESET;
                            else if (i < bar_cells * 0.8)
                                bar += BRIGHT_CYAN  + std::string("█") + RESET;
                            else
                                bar += BRIGHT_BLUE  + std::string("█") + RESET;
                        }
                        else if (i == bar_filled && fraction < 1.0)
                            bar += BRIGHT_WHITE + std::string("▌") + RESET;
                        else
                            bar += DIM + std::string("░") + RESET;
                    }

                    std::cout << oss_prefix.str() << bar << suffix << std::flush;
                    break;
                }
            }
        }

        // ── Propose new parameters ──
        Subs::Array1D<double> prop = current_pars;
        for (int i = 0; i < npar; ++i) {
            double proposal = current_pars[i] + dsteps[i] * gauss(rng);

            // Reflect off boundaries until in range
            while (proposal < limits[i].first || proposal > limits[i].second) {
                if (proposal < limits[i].first)
                    proposal = 2*limits[i].first - proposal;
                else if (proposal > limits[i].second)
                    proposal = 2*limits[i].second - proposal;
            }
            prop[i] = proposal;
        }
        model.set_param(prop);

        // Calculate log prior for proposed parameters
        double log_prior_prop = 0.0;
        if (use_priors) {
            double prop_inclination = model.iangle.vary && iangle_idx >= 0 ? prop[iangle_idx] : model.iangle.value;
            double prop_q = model.q.vary && q_idx >= 0 ? prop[q_idx] : model.q.value;
            double prop_vs = model.velocity_scale.vary && vs_idx >= 0 ? prop[vs_idx] : model.velocity_scale.value;
            double prop_r1 = model.r1.vary && r1_idx >= 0 ? prop[r1_idx] : model.r1.value;

            log_prior_prop += log_mass_ratio_pdf(prop_inclination, prop_q, prop_vs, prop_r1);

            if (!prior_warning_printed) {
                if (!model.q.vary && q_idx < 0) {
                    cout << "[WARNING] Priors are set but q is fixed, value may not be physically sensible!" << endl;
                    prior_warning_printed = true;
                }
                if (!model.velocity_scale.vary && vs_idx < 0) {
                    cout << "[WARNING] Priors are set but velocity_scale is fixed, value may not be physically sensible!" << endl;
                    prior_warning_printed = true;
                }
                if (!model.r1.vary && r1_idx < 0) {
                    cout << "[WARNING] Priors are set but r1 is fixed, value may not be physically sensible!" << endl;
                    prior_warning_printed = true;
                }
            }
        }

        // ── Evaluate proposed model ──
        vector<double> fitp;
        double wdp, chp, wnp, lg1p, lg2p, rv1p, rv2p;
        bool computed_model = false;

        if (use_grid_cache) {
            vector<double> prop_vec(prop.begin(), prop.end());
            auto grid_idx = grid_cache->param_to_grid_index(prop_vec);
            auto grid_params = grid_cache->grid_index_to_param(grid_idx);

            GridCache::CachedModel cached_model;

            if (grid_cache->is_close_to_grid_point(prop_vec, grid_threshold) &&
                grid_cache->get_cached_model(grid_idx, cached_model)) {
                chp = cached_model.chisq;
                fitp = cached_model.model_values;
                computed_model = false;
            } else {
                if (!grid_cache->has_node(grid_idx)) {
                    Subs::Array1D<double> grid_array(npar);
                    for (int i = 0; i < npar; ++i) {
                        grid_array[i] = grid_params[i];
                    }
                    model.set_param(grid_array);

                    try {
                        vector<double> grid_fit;
                        double grid_chisq;
                        light_curve_comp_fast(model, data, scale, !no_file, false, sfac,
                                            grid_fit, wdp, grid_chisq, wnp,
                                            lg1p, lg2p, rv1p, rv2p, max_model_points);

                        GridCache::CachedModel new_model;
                        new_model.model_values = grid_fit;
                        new_model.chisq = grid_chisq;
                        grid_cache->add_model(grid_idx, new_model);
                    } catch (Lcurve::Lcurve_Error &e) {
                        GridCache::CachedModel bad_model;
                        bad_model.chisq = 1e10;
                        grid_cache->add_model(grid_idx, bad_model);
                    }
                }

                model.set_param(prop);
                try {
                    light_curve_comp_fast(model, data, scale, !no_file, false, sfac,
                                        fitp, wdp, chp, wnp,
                                        lg1p, lg2p, rv1p, rv2p, max_model_points);
                    computed_model = true;
                } catch (Lcurve::Lcurve_Error &e) {
                    model.set_param(current_pars);
                    record_adapt(false);   // count as rejection for adaptation
                    if (step >= burn_in) {
                        chain[step-burn_in] = ChainEntry{step-burn_in, current_pars, current_chisq};
                    }
                    continue;
                }
            }

        } else {
            try {
                light_curve_comp_fast(model, data, scale, !no_file, false, sfac,
                                     fitp, wdp, chp, wnp,
                                     lg1p, lg2p, rv1p, rv2p, max_model_points);
                computed_model = true;
            }
            catch (Lcurve::Lcurve_Error &e) {
                model.set_param(current_pars);
                record_adapt(false);   // count as rejection for adaptation
                if (step >= burn_in) {
                    chain[step-burn_in] = ChainEntry{step-burn_in, current_pars, current_chisq};
                }
                continue;
            }
        }

        // For visualization, ensure we have the model
        if (step % progress_interval == 0 && !computed_model) {
            model.set_param(prop);
            light_curve_comp_fast(model, data, scale, !no_file, false, sfac,
                                fitp, wdp, chp, wnp,
                                lg1p, lg2p, rv1p, rv2p, max_model_points);
        }

        // ── Accept / reject ──
        bool this_accept = false;
        double log_alpha = -0.5 * (chp - current_chisq) + (log_prior_prop - log_prior_current);

        if (log_alpha >= 0.0 || log(uni(rng)) < log_alpha) {
            current_pars = prop;
            log_prior_current = log_prior_prop;
            current_chisq = chp;
            ++accepted;
            this_accept = true;
            if (chp < best_chisq) {
                best_chisq = chp;
                best_pars   = prop;
            }
        } else {
            model.set_param(current_pars);
        }

        // ── Feed the adaptation machinery ──
        record_adapt(this_accept);

        // ── Sliding-window update & chain storage (post-burn-in) ──
        if (step >= burn_in) {
            acc_window.push_back(this_accept);
            if (this_accept) ++window_accept_count;
            if (int(acc_window.size()) > window_size) {
                if (acc_window.front()) --window_accept_count;
                acc_window.pop_front();
            }
            chain[step-burn_in] = ChainEntry{step-burn_in, current_pars, current_chisq};
        }

        if (step % progress_interval == 0) {
            Helpers::plot_model_live(data, fitp, no_file, copy, gp);
        }
    }

    // Finish bar
    auto t_end = Clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);

    auto total_seconds = duration.count() / 1000.0;
    int hours = static_cast<int>(total_seconds / 3600);
    int minutes = static_cast<int>((total_seconds - hours * 3600) / 60);
    double seconds = total_seconds - hours * 3600 - minutes * 60;

    cout << "\n" << BRIGHT_GREEN << "✓ MCMC sampling completed! Took: ";
    if (hours > 0) {
        cout << hours << "h ";
    }
    if (minutes > 0) {
        cout << minutes << "m ";
    }
    cout << fixed << setprecision(2) << seconds << "s" << RESET << endl;

    // ── Final adaptation report ──
    if (adapt_enabled) {
        double final_sf = std::exp(adapt_log_scale);
        double final_gamma = adapt_rate * std::pow(1.0 + adapt_batch, -adapt_decay);
        cout << BRIGHT_CYAN << "Final adaptation state:" << RESET << endl;
        cout << "  Total batches:    " << adapt_batch << endl;
        cout << "  Final γ:          " << scientific << setprecision(2)
             << final_gamma << endl;
        cout << "  Final scale:      ×" << fixed << setprecision(4)
             << final_sf << endl;
        cout << "  Last batch rate:  " << setprecision(1)
             << (adapt_current_rate * 100.0) << "%"
             << " (target " << (adapt_target * 100.0) << "%)" << endl;
    }

    // Report
    cout << "Best chi^2 = " << best_chisq << endl;
    for (int i = 0; i < npar; ++i) cout << names[i] << " = " << best_pars[i] << endl;

    // Write chain to file
    ofstream chain_file(config.value("chain_out_path", "chain_out.txt"));
    chain_file << "step";
    for (auto &n : names) chain_file << "," << n;
    chain_file << ",chisq\n";
    for (auto &entry : chain) {
        chain_file << entry.step;
        for (int i = 0; i < npar; ++i) chain_file << "," << entry.pars[i];
        chain_file << "," << entry.chisq << "\n";
    }
    chain_file.close();

    // Best-fit light curve & plot etc.
    vector<double> best_fit;
    model.set_param(best_pars);
    Lcurve::light_curve_comp(model, data, scale, !no_file, false, sfac,
                             best_fit, wd0, chisq0, wn0,
                             lg10, lg20, rv10, rv20);
    if (device != "none" && device != "null") Helpers::plot_model(data, best_fit, no_file, copy, device);

    // Write output file
    string sout = config["output_file_path"].get<string>();
    for (long unsigned int i = 0; i < data.size(); ++i) data[i].flux = best_fit[i] + noise * Subs::gauss2(seed);
    Helpers::write_data(data, sout);

    // ── Persist adaptation results so future runs can start from these stepsizes ──
    if (adapt_enabled) {
        json adapt_out;
        adapt_out["final_scale_factor"]         = std::exp(adapt_log_scale);
        adapt_out["last_batch_acceptance_rate"]  = adapt_current_rate;
        adapt_out["target_acceptance_rate"]      = adapt_target;
        adapt_out["total_batches"]               = adapt_batch;
        adapt_out["final_learning_rate"]         = adapt_rate * std::pow(1.0 + adapt_batch, -adapt_decay);
        for (int i = 0; i < npar; ++i) {
            adapt_out["initial_stepsizes"][names[i]] = initial_dsteps[i];
            adapt_out["adapted_stepsizes"][names[i]] = dsteps[i];
        }
        config["adaptation_results"] = adapt_out;
    }
    Helpers::write_config_and_model_to_json(model, config,
        config["output_file_path"].get<string>() + ".json");

    return 0;
}