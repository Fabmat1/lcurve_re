#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <cmath>
#include <vector>
#include <deque>
#include <chrono>
#include <nlohmann/json.hpp>
#include "../src/lcurve_base/lcurve.h"
#include "../src/new_helpers.h"
#include "../src/new_subs.h"

using namespace std;
using json = nlohmann::json;
using Clock = chrono::steady_clock;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <config_file.json>" << endl;
        return 1;
    }
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
    Subs::Buffer1D<double> sfac;
    Helpers::load_seed_scale_sfac(config, no_file, model, seed, scale, sfac);

    // Use Priors if available


    // Get variable parameters
    int npar = model.nvary();
    vector<string> names(npar);
    for (int i = 0; i < npar; ++i) names[i] = model.get_name(i);
    Subs::Array1D<double> current_pars = model.get_param();
    Subs::Array1D<double> dsteps      = model.get_dstep();
    vector<pair<double, double>> limits = model.get_limit();
    string device = config.value("plot_device", "none");

    // Define Gnuplot instance
    Gnuplot gp; // Open once, reuse
    gp << "set terminal " + device + " title 'Live fitting plot'\n";  // no 'persist'
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
    int bar_width         = config.value("progress_bar_width", 50);

    // prepare sliding-window acceptance-rate over last 10% of post-burn-in
    int post_len = nsteps - burn_in;
    int window_size = max(1, int(post_len * 0.1));    // last 10%
    deque<bool> acc_window;
    int window_accept_count = 0;

    // RNG
    mt19937 rng(seed);
    normal_distribution<> gauss(0.0, 1.0);
    uniform_real_distribution<> uni(0.0, 1.0);

    // Initial chi-squared
    Subs::Array1D<double> fit;
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

    // Timing
    auto t_start = Clock::now();
    int accepted = 0;

    tuple<double, double, double> vobs_prior;
    tuple<double, double, double> m1_prior;
    tuple<double, double, double> m2_min_prior;
    tuple<double, double, double> r1_prior;
    bool prior_warning_printed = false;

    // Stuff related to priors
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
    }


    // MCMC
    for (int step = 0; step < nsteps; ++step) {
        // Progress bar and ETA
        if (step % progress_interval == 0) {
            double fraction = double(step) / nsteps;
            int pos = int(bar_width * fraction);
            auto now = Clock::now();
            double elapsed = chrono::duration<double>(now - t_start).count();
            double est_total = elapsed / max(fraction, 1e-8);
            double eta = est_total - elapsed;
            int eta_i = int(eta);
            int h = eta_i / 3600;
            int m = (eta_i % 3600) / 60;
            int s = eta_i % 60;
            string eta_str;
            if (h > 0) eta_str = to_string(h) + "h " + to_string(m) + "m " + to_string(s) + "s";
            else if (m > 0) eta_str = to_string(m) + "m " + to_string(s) + "s";
            else eta_str = to_string(s) + "s";

            // compute acceptance rate over sliding window
            double acc_rate = 0.0;
            if (!acc_window.empty()) {
                acc_rate = double(window_accept_count) / acc_window.size() * 100.0;
            }
            else acc_rate = -10.0;

            cout << "\r[";
            for (int i = 0; i < bar_width; ++i) cout << (i < pos ? "█" : " ");
            if (acc_rate >= 0) {
                cout << "] " << int(fraction*100) << "% "
                     << "Acc " <<  int(acc_rate)  << "% "
                     << "ETA " << eta_str;
            }
            else {
                cout << "] " << int(fraction*100) << "% "
                     << "Acc not available during burn-in! "
                     << "ETA " << eta_str;
            }
            cout.flush();
        }

        // Propose
        Subs::Array1D<double> prop = current_pars;
        for (int i = 0; i < npar; ++i) {
            double proposal = current_pars[i] + dsteps[i] * gauss(rng);
            proposal = max(limits[i].first, min(limits[i].second, proposal));
            prop[i] = proposal;
        }
        model.set_param(prop);

        if (use_priors) {
            double current_inclination;
            if (model.iangle.vary) {
                auto it = find(names.begin(), names.end(), "iangle");
                int iangle_idx = (it != names.end()) ? distance(names.begin(), it) : -1;
                current_inclination = prop[iangle_idx];
            }
            else {
                current_inclination = model.iangle.value;
            }

            double current_q;
            // See if mass ratio is plausible
            if (model.q.vary) {
                double q_min = Helpers::mass_ratio_from_inclination(current_inclination,
                    get<0>(m1_prior)+get<1>(m1_prior),
                    get<0>(m2_min_prior)-get<2>(m2_min_prior));
                double q_max = Helpers::mass_ratio_from_inclination(current_inclination,
                    get<0>(m1_prior)-get<2>(m1_prior),
                    get<0>(m2_min_prior)+get<1>(m2_min_prior));

                auto it = find(names.begin(), names.end(), "q");
                int q_idx = (it != names.end()) ? distance(names.begin(), it) : -1;

                if (prop[q_idx] < q_min || prop[q_idx] > q_max) {
                    if (step < 50)
                    {
                        cout << "Proposed q is outside of mass ratio prior range, rejecting!" << endl;
                        cout << "Proposed q = " << prop[q_idx] << endl;
                        cout << "q_min = " << q_min << endl;
                        cout << "q_max = " << q_max << endl;
                    }
                    model.set_param(current_pars);

                    if (step >= burn_in) {
                        acc_window.push_back(false);
                        if (int(acc_window.size()) > window_size) {
                            if (acc_window.front()) --window_accept_count;
                            acc_window.pop_front();
                        }
                        chain[step-burn_in] = ChainEntry{step-burn_in, current_pars, current_chisq};
                    }
                    continue;
                }
                else {
                    current_q = prop[q_idx];
                }
            }
            else {
                if (not prior_warning_printed) {
                    cout << "[WARNING] Priors are set but q is fixed, value may not be physically sensible!" << endl;
                    prior_warning_printed = true;                
                }
                current_q = model.q.value;
            }

            double current_vscale;

            // See if velocity scale is plausible
            if (model.velocity_scale.vary) {
                double vs_min = Helpers::velocity_scale_from_inclination(current_inclination,
                    get<0>(vobs_prior)-get<2>(vobs_prior), current_q);
                double vs_max = Helpers::velocity_scale_from_inclination(current_inclination,
                    get<0>(vobs_prior)+get<1>(vobs_prior), current_q);

                auto it = find(names.begin(), names.end(), "velocity_scale");
                int vs_idx = (it != names.end()) ? distance(names.begin(), it) : -1;

                if (prop[vs_idx] < vs_min || prop[vs_idx] > vs_max) {
                    if (step < 50)
                    {
                        cout << "Proposed velocity scale is outside of velocity scale prior range, rejecting!" << endl;
                        cout << "Proposed velocity scale = " << prop[vs_idx] << endl;
                        cout << "vs_min = " << vs_min << endl;
                        cout << "vs_max = " << vs_max << endl;
                    }

                    model.set_param(current_pars);

                    if (step >= burn_in) {
                        acc_window.push_back(false);
                        if (int(acc_window.size()) > window_size) {
                            if (acc_window.front())
                                --window_accept_count;
                            acc_window.pop_front();
                        }
                        chain[step - burn_in] = ChainEntry{step - burn_in, current_pars, current_chisq};
                    }
                    continue;
                }
                current_vscale = prop[vs_idx];
            }
            else {
                if (not prior_warning_printed) {
                    cout << "[WARNING] Priors are set but velocity scale is fixed, value may not be physically sensible!" << endl;
                    prior_warning_printed = true;
                }
                current_vscale = model.velocity_scale.value;
            }

            // See if R1 is plausible
            if (model.r1.vary) {
                double true_period = config["true_period"].get<double>();
                double r1_min = Helpers::compute_scaled_r1(get<0>(r1_prior)-get<2>(r1_prior), current_vscale, true_period);
                double r1_max = Helpers::compute_scaled_r1(get<0>(r1_prior)+get<1>(r1_prior), current_vscale, true_period);

                auto it = find(names.begin(), names.end(), "r1");
                int r1_idx = (it != names.end()) ? distance(names.begin(), it) : -1;

                if (prop[r1_idx] < r1_min || prop[r1_idx] > r1_max) {
                    if (step < 50)
                    {
                        cout << "Proposed R1 is outside of R1 prior range, rejecting!" << endl;
                        cout << "Proposed R1 = " << prop[r1_idx] << endl;
                        cout << "r1_min = " << r1_min << endl;
                        cout << "r1_max = " << r1_max << endl;
                    }


                    model.set_param(current_pars);

                    if (step >= burn_in) {
                        acc_window.push_back(false);
                        if (int(acc_window.size()) > window_size) {
                            if (acc_window.front())
                                --window_accept_count;
                            acc_window.pop_front();
                        }
                        chain[step - burn_in] = ChainEntry{step - burn_in, current_pars, current_chisq};
                    }
                    continue;
                }
            }
            else {
                if (not prior_warning_printed) {
                    cout << "[WARNING] Priors are set but R1 is fixed, value may not be physically sensible!" << endl;
                    prior_warning_printed = true;
                }
            }
        }


        // Evaluate
        Subs::Array1D<double> fitp;
        double wdp, chp, wnp, lg1p, lg2p, rv1p, rv2p;
        try {
            light_curve_comp(model, data, scale, !no_file, false, sfac,
                                     fitp, wdp, chp, wnp,
                                     lg1p, lg2p, rv1p, rv2p);
        }
        catch (Lcurve::Lcurve_Error &e) {
            model.set_param(current_pars);
            if (step >= burn_in) {
                chain[step-burn_in] = ChainEntry{step-burn_in, current_pars, current_chisq};
            }
            continue;
        };

        // Accept/reject
        bool this_accept = false;
        double alpha = exp(-(chp - current_chisq)/2.0);
        if (alpha >= 1.0 || uni(rng) < alpha) {
            current_pars = prop;
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

        // sliding-window update & chain storage
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
    cout << "\r[";
    for (int i = 0; i < bar_width; ++i) cout << "█";
    cout << "] 100% Completed in "
         << int(chrono::duration<double>(t_end - t_start).count())
         << "s" << endl;

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
    Subs::Array1D<double> best_fit;
    model.set_param(best_pars);
    Lcurve::light_curve_comp(model, data, scale, !no_file, false, sfac,
                             best_fit, wd0, chisq0, wn0,
                             lg10, lg20, rv10, rv20);
    if (device != "none" && device != "null") Helpers::plot_model(data, best_fit, no_file, copy, device);

    // Write output file
    string sout = config["output_file_path"].get<string>();
    for (int i = 0; i < data.size(); ++i) data[i].flux = best_fit[i] + noise * Subs::gauss2(seed);
    Helpers::write_data(data, sout);

    return 0;
}
