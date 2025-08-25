#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <cmath>
#include <vector>
#include <iomanip>
#include <deque>
#include <chrono>
#include <nlohmann/json.hpp>
#include "../src/lcurve_base/lcurve.h"
#include "../src/new_helpers.h"
#include "../src/new_subs.h"
#include "../src/mass_ratio_pdf.h"
#include <cmath>

using namespace std;
using json = nlohmann::json;
using Clock = chrono::steady_clock;

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
    vector<double> sfac;
    Helpers::load_seed_scale_sfac(config, no_file, model, seed, scale, sfac);

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

    // ANSI color codes
    const string RESET = "\033[0m";
    const string BRIGHT_GREEN = "\033[92m";
    const string BRIGHT_BLUE = "\033[94m";
    const string BRIGHT_YELLOW = "\033[93m";
    const string BRIGHT_RED = "\033[91m";
    const string BRIGHT_CYAN = "\033[96m";
    const string BRIGHT_WHITE = "\033[97m";
    const string DIM = "\033[2m";

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

    // Add these variable declarations before the MCMC loop
    int q_idx = -1;
    int vs_idx = -1;
    int r1_idx = -1;
    int iangle_idx = -1;
    
    // Find indices for the parameters we need
    for (int i = 0; i < npar; ++i) {
        if (names[i] == "q") q_idx = i;
        else if (names[i] == "velocity_scale") vs_idx = i;
        else if (names[i] == "r1") r1_idx = i;
        else if (names[i] == "iangle") iangle_idx = i;
    }

    double log_prior_current = 0.0;
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
        
        // Initialize mass ratio PDF grid
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
                                       config.value("true_period", 1.0), 0.0);

        // Calculate initial log prior
        double init_incl = model.iangle.vary && iangle_idx >= 0 ? current_pars[iangle_idx] : model.iangle.value;
        double init_q = model.q.vary && q_idx >= 0 ? current_pars[q_idx] : model.q.value;
        double init_vs = model.velocity_scale.vary && vs_idx >= 0 ? current_pars[vs_idx] : model.velocity_scale.value;
        double init_r1 = model.r1.vary && r1_idx >= 0 ? current_pars[r1_idx] : model.r1.value;
        
        // Mass ratio prior
        log_prior_current += log_mass_ratio_pdf(init_incl, init_q, init_vs, init_r1);
    }

    // Timing
    auto t_start = Clock::now();
    int accepted = 0;

    // MCMC
    for (int step = 0; step < nsteps; ++step) {
        // Enhanced Progress bar with colors and better formatting
        if (step % progress_interval == 0) {
            double fraction = double(step) / nsteps;
            int pos = int(bar_width * fraction);
            
            auto now = Clock::now();
            double elapsed = chrono::duration<double>(now - t_start).count();
            double est_total = elapsed / max(fraction, 1e-8);
            double eta = est_total - elapsed;
            
            // Format ETA with better time display
            int eta_i = int(eta);
            int h = eta_i / 3600;
            int m = (eta_i % 3600) / 60;
            int s = eta_i % 60;
            
            string eta_str;
            if (h > 0) {
                eta_str = string(h < 10 ? "0" : "") + to_string(h) + ":" + 
                         string(m < 10 ? "0" : "") + to_string(m) + ":" + 
                         string(s < 10 ? "0" : "") + to_string(s);
            } else if (m > 0) {
                eta_str = string(m < 10 ? "0" : "") + to_string(m) + ":" + 
                         string(s < 10 ? "0" : "") + to_string(s);
            } else {
                eta_str = to_string(s) + "s";
            }
            
            // Compute acceptance rate over sliding window
            double acc_rate = 0.0;
            if (!acc_window.empty()) {
                acc_rate = double(window_accept_count) / acc_window.size() * 100.0;
            } else {
                acc_rate = -10.0;
            }
            
            // Progress bar with gradient effect
            cout << "\r" << BRIGHT_WHITE << "MCMC " << RESET;
            cout << "[";
            
            for (int i = 0; i < bar_width; ++i) {
                if (i < pos) {
                    // Gradient effect: green to blue
                    if (i < pos * 0.6) {
                        cout << BRIGHT_GREEN << "█" << RESET;
                    } else if (i < pos * 0.8) {
                        cout << BRIGHT_CYAN << "█" << RESET;
                    } else {
                        cout << BRIGHT_BLUE << "█" << RESET;
                    }
                } else if (i == pos && fraction < 1.0) {
                    // Show current position with a different character
                    cout << BRIGHT_WHITE << "▌" << RESET;
                } else {
                    cout << DIM << "░" << RESET;
                }
            }
            
            cout << "] ";
            
            // Progress percentage with color coding
            int percent = int(fraction * 100);
            if (percent >= 90) {
                cout << BRIGHT_GREEN << percent << "%" << RESET;
            } else if (percent >= 50) {
                cout << BRIGHT_BLUE << percent << "%" << RESET;
            } else {
                cout << BRIGHT_YELLOW << percent << "%" << RESET;
            }
            
            // Acceptance rate with color coding
            if (acc_rate >= 0) {
                cout << " │ ";
                if (acc_rate >= 20 && acc_rate <= 50) {
                    cout << BRIGHT_GREEN << "Acc " << fixed << setprecision(1) << acc_rate << "%" << RESET;
                } else if (acc_rate < 20) {
                    cout << BRIGHT_RED << "Acc " << fixed << setprecision(1) << acc_rate << "%" << RESET;
                } else {
                    cout << BRIGHT_YELLOW << "Acc " << fixed << setprecision(1) << acc_rate << "%" << RESET;
                }
            } else {
                cout << " │ " << DIM << "Acc burn-in" << RESET;
            }
            
            // ETA with icon
            cout << " │ " << BRIGHT_CYAN << "⏱ " << eta_str << RESET;
            
            // Add iteration info
            cout << " │ " << DIM << step << "/" << nsteps << RESET;
            
            // Add some padding to clear any leftover characters
            cout << "    ";
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
        
        // Calculate log prior for proposed parameters
        double log_prior_prop = 0.0;
        if (use_priors) {
            double prop_inclination = model.iangle.vary && iangle_idx >= 0 ? prop[iangle_idx] : model.iangle.value;
            double prop_q = model.q.vary && q_idx >= 0 ? prop[q_idx] : model.q.value;
            double prop_vs = model.velocity_scale.vary && vs_idx >= 0 ? prop[vs_idx] : model.velocity_scale.value;
            double prop_r1 = model.r1.vary && r1_idx >= 0 ? prop[r1_idx] : model.r1.value;
            
            // Mass ratio prior
            log_prior_prop += log_mass_ratio_pdf(prop_inclination, prop_q, prop_vs, prop_r1);

            // Print warnings if parameters are fixed but priors are set
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
        
        // Evaluate
        vector<double> fitp;
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
        }


        // Accept/reject
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
               best_pars = prop;
           }
       } else {
           model.set_param(current_pars);
       }

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
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);

    // Convert to hours, minutes, seconds
    auto total_seconds = duration.count() / 1000.0;
    int hours = static_cast<int>(total_seconds / 3600);
    int minutes = static_cast<int>((total_seconds - hours * 3600) / 60);
    double seconds = total_seconds - hours * 3600 - minutes * 60;

    // Don't forget to add a newline when sampling is complete
    cout << "\n" << BRIGHT_GREEN << "✓ MCMC sampling completed! Took: ";

    // Only print hours if != 0
    if (hours > 0) {
        cout << hours << "h ";
    }

    // Only print minutes if != 0
    if (minutes > 0) {
        cout << minutes << "m ";
    }

    cout << fixed << setprecision(2) << seconds << "s" << RESET << endl;

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

    return 0;
}
