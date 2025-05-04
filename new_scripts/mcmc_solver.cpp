// fit_mcmc.cpp
// Uses Metropolis–Hastings MCMC to optimize light‐curve model parameters

#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <cmath>
#include <vector>
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

    // Get variable parameters
    int npar = model.nvary();
    vector<string> names(npar);
    for (int i = 0; i < npar; ++i) names[i] = model.get_name(i);
    Subs::Array1D<double> current_pars = model.get_param();
    Subs::Array1D<double> dsteps      = model.get_dstep();
    vector<pair<double, double>> limits = model.get_limit();

    cout << "Calculating MCMC for " << npar << " parameters:" << endl;
    for (int i = 0; i < npar; ++i) {
        cout << names[i] << ": " << current_pars[i] << " with stepsize " << dsteps[i] << " and limits from " << limits[i].first << " to " << limits[i].second << endl;
    }

    // MCMC settings
    int nsteps            = config.value("mcmc_steps", 1000);
    int burn_in           = config.value("mcmc_burn_in", nsteps/4);
    int progress_interval = config.value("progress_interval", 50);
    int bar_width         = config.value("progress_bar_width", 50);

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
            double acc_rate = step > 0 ? double(accepted) / step * 100.0 : 0.0;
            cout << "\r[";
            for (int i = 0; i < bar_width; ++i) cout << (i < pos ? "█" : " ");
            cout << "] " << int(fraction*100) << "% "
                 << "Acc " << int(acc_rate) << "% "
                 << "ETA " << eta_str;
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

        // Evaluate
        Subs::Array1D<double> fitp;
        double wdp, chp, wnp, lg1p, lg2p, rv1p, rv2p;
        try {
            Lcurve::light_curve_comp(model, data, scale, !no_file, false, sfac,
                                     fitp, wdp, chp, wnp,
                                     lg1p, lg2p, rv1p, rv2p);
        }
        except Lcurve::Lcurve_Error &e {
            model.set_param(current_pars);
            // Store post-burn-in entries in memory
            if (step >= burn_in) {
                chain[step-burn_in] = ChainEntry({step-burn_in, current_pars, current_chisq});
            }
            continue;
        };

        // Accept/reject
        double alpha = exp(-(chp - current_chisq)/2.0);
        if (alpha >= 1.0 || uni(rng) < alpha) {
            current_pars = prop;
            current_chisq = chp;
            ++accepted;
            if (chp < best_chisq) {
                best_chisq = chp;
                best_pars   = prop;
            }
        } else {
            model.set_param(current_pars);
        }

        // Store post-burn-in entries in memory
        if (step >= burn_in) {
            chain[step-burn_in] = ChainEntry({step-burn_in, current_pars, current_chisq});
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

    // Write chain to file at once
    ofstream chain_file("chain_out.txt");
    chain_file << "step";
    for (auto &n : names) chain_file << "," << n;
    chain_file << ",chisq\n";
    for (auto &entry : chain) {
        chain_file << entry.step;
        for (int i = 0; i < npar; ++i) chain_file << "," << entry.pars[i];
        chain_file << "," << entry.chisq << "\n";
    }
    chain_file.close();

    // Best-fit light curve
    Subs::Array1D<double> best_fit;
    model.set_param(best_pars);
    Lcurve::light_curve_comp(model, data, scale, !no_file, false, sfac,
                             best_fit, wd0, chisq0, wn0,
                             lg10, lg20, rv10, rv20);

    // Plot
    string device = config.value("plot_device", "none");
    if (device != "none" && device != "null") Helpers::plot_model(data, best_fit, no_file, copy, device);

    // Write
    string sout = config["output_file_path"].get<string>();
    for (int i = 0; i < data.size(); ++i) data[i].flux = best_fit[i] + noise * Subs::gauss2(seed);
    Helpers::write_data(data, sout);

    return 0;
}
