#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <nlohmann/json.hpp>
#include "../src/lcurve_base/lcurve.h"
#include "../src/new_helpers.h"
#include "../src/new_subs.h"

using namespace std;
using json = nlohmann::json;
using Clock = chrono::steady_clock;

// Objective function: compute chi-squared for given parameters
static double compute_chisq(Lcurve::Model &model, const Lcurve::Data &data,
                            bool scale, bool have_data,
                            vector<double> &sfac) {
    vector<double> fit;
    double wd, chisq, wn, lg1, lg2, rv1, rv2;
    Lcurve::light_curve_comp(model, data, scale, have_data, false, sfac,
                             fit, wd, chisq, wn, lg1, lg2, rv1, rv2);
    return chisq;
}

// Nelder-Mead Simplex algorithm
vector<double> nelder_mead(function<double(const vector<double>&)> f,
                           const vector<double> &x0,
                           const vector<double> &step,
                           const vector<pair<double,double>> &limits,
                           int max_iter = 999999,
                           double tol = 5e-5) {
    int n = x0.size();
    int simplex_size = n + 1;
    vector<vector<double>> simplex(simplex_size, x0);
    // initial simplex
    for (int i = 0; i < n; ++i) {
        simplex[i+1][i] = x0[i] + step[i];
        // enforce limits
        simplex[i+1][i] = min(max(simplex[i+1][i], limits[i].first), limits[i].second);
    }
    vector<double> fvals(simplex_size);
    for (int i = 0; i < simplex_size; ++i) fvals[i] = f(simplex[i]);

    for (int iter = 0; iter < max_iter; ++iter) {
        if (iter % 10 == 0) {
            cout << "Iteration " << iter << " ";
            cout << " Weighted Chi^2 = " << accumulate(fvals.begin(), fvals.end(), 0.0)/n << endl;
        };
        // sort by fvals
        vector<int> idx(simplex_size);
        iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(), [&](int a, int b) { return fvals[a] < fvals[b]; });
        // best is idx[0], worst is idx[n]
        vector<double> x_best = simplex[idx[0]];
        vector<double> x_worst = simplex[idx[n]];
        // compute centroid of all but worst
        vector<double> x_centroid(n, 0.0);
        for (int i = 0; i < simplex_size; ++i) {
            if (i == idx[n]) continue;
            for (int j = 0; j < n; ++j) x_centroid[j] += simplex[i][j];
        }
        for (double &val : x_centroid) val /= n;

        // reflection
        double alpha = 1.0;
        vector<double> x_ref(n);
        for (int j = 0; j < n; ++j)
            x_ref[j] = x_centroid[j] + alpha*(x_centroid[j] - x_worst[j]);
        // enforce limits
        for (int j = 0; j < n; ++j)
            x_ref[j] = min(max(x_ref[j], limits[j].first), limits[j].second);
        double f_ref = f(x_ref);

        if (f_ref < fvals[idx[0]]) {
            // expansion
            double gamma = 2.0;
            vector<double> x_exp(n);
            for (int j = 0; j < n; ++j)
                x_exp[j] = x_centroid[j] + gamma*(x_ref[j] - x_centroid[j]);
            for (int j = 0; j < n; ++j)
                x_exp[j] = min(max(x_exp[j], limits[j].first), limits[j].second);
            double f_exp = f(x_exp);
            if (f_exp < f_ref) {
                simplex[idx[n]] = x_exp;
                fvals[idx[n]] = f_exp;
            } else {
                simplex[idx[n]] = x_ref;
                fvals[idx[n]] = f_ref;
            }
        } else if (f_ref < fvals[idx[n-1]]) {
            simplex[idx[n]] = x_ref;
            fvals[idx[n]] = f_ref;
        } else {
            // contraction
            double rho = 0.5;
            vector<double> x_con(n);
            for (int j = 0; j < n; ++j)
                x_con[j] = x_centroid[j] + rho*(x_worst[j] - x_centroid[j]);
            for (int j = 0; j < n; ++j)
                x_con[j] = min(max(x_con[j], limits[j].first), limits[j].second);
            double f_con = f(x_con);
            if (f_con < fvals[idx[n]]) {
                simplex[idx[n]] = x_con;
                fvals[idx[n]] = f_con;
            } else {
                // shrink
                double sigma = 0.5;
                for (int i = 1; i < simplex_size; ++i) {
                    for (int j = 0; j < n; ++j)
                        simplex[i][j] = simplex[idx[0]][j] + sigma*(simplex[i][j] - simplex[idx[0]][j]);
                    for (int j = 0; j < n; ++j)
                        simplex[i][j] = min(max(simplex[i][j], limits[j].first), limits[j].second);
                    fvals[i] = f(simplex[i]);
                }
            }
        }
        // check convergence
        double f_mean = accumulate(fvals.begin(), fvals.end(), 0.0)/simplex_size;
        double var = 0.0;
        for (double v : fvals) var += (v - f_mean)*(v - f_mean);
        var /= simplex_size;
        if (sqrt(var) < tol) break;
    }

    // return best point
    int best_idx = min_element(fvals.begin(), fvals.end()) - fvals.begin();
    return simplex[best_idx];
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <config_file.json>" << endl;
        return 1;
    }
    // Load model and config
    string config_file = argv[1];
    auto model_config = Helpers::load_model_and_config_from_json(config_file);
    Lcurve::Model model = model_config.first;
    json config = model_config.second;

    // Load data
    auto data_copy = Helpers::read_and_copy_lightcurve_from_file(config["data_file_path"]);
    Lcurve::Data data = data_copy.first;
    Lcurve::Data copy = data_copy.second;
    if (data.empty()) throw Lcurve::Lcurve_Error("No data file provided");
    double noise = config["noise"].get<double>();

    // Initialize scales etc.
    int seed;
    bool scale;
    vector<double> sfac;
    Helpers::load_seed_scale_sfac(config, false, model, seed, scale, sfac);

    // Get variable parameters
    int npar = model.nvary();
    Subs::Array1D<double> init_pars = model.get_param();
    Subs::Array1D<double> dsteps   = model.get_dstep();
    auto limits = model.get_limit();

    // Prepare initial guess and step vectors
    vector<double> x0(npar), steps(npar);
    for (int i = 0; i < npar; ++i) {
        x0[i] = init_pars[i];
        steps[i] = dsteps[i];
    }

    // Define objective wrapper
    auto obj = [&](const vector<double> &pars) {
        Subs::Array1D<double> arr(npar);
        for (int i = 0; i < npar; ++i) arr[i] = pars[i];
        model.set_param(arr);
        return compute_chisq(model, data, scale, true, sfac);
    };

    // Run Simplex
    auto t0 = Clock::now();
    vector<double> best = nelder_mead(obj, x0, steps, limits,
                                      config.value("simplex_max_iter", 200),
                                      config.value("simplex_tol", 1e-6));
    // Ensure returned vector matches parameter count
    if(best.size() != x0.size()) {
        best.resize(x0.size());
    }
    double best_chi = obj(best);
    auto t1 = Clock::now();


    // Report
    cout << "Simplex optimization completed in "
         << chrono::duration<double>(t1 - t0).count() << "s" << endl;
    cout << "Best chi^2 = " << best_chi << endl;
    for (int i = 0; i < npar; ++i) cout << model.get_name(i)
                                        << " = " << best[i] << endl;

    // Write output parameters, fit, and optionally plot
    Subs::Array1D<double> best_arr(npar);
    vector<double> best_fit;
    for (int i = 0; i < npar; ++i) best_arr[i] = best[i];
    model.set_param(best_arr);
    // Prepare placeholders for unused outputs
    double wd_out, chisq_out, wn_out, lg1_out, lg2_out, rv1_out, rv2_out;
    Lcurve::light_curve_comp(model, data, scale, true, false, sfac,
                             best_fit,
                             wd_out, chisq_out,
                             wn_out, lg1_out,
                             lg2_out, rv1_out,
                             rv2_out);
    // Save chain-less results
    ofstream out(config["output_file_path"].get<string>());
    for (size_t i = 0; i < data.size(); ++i) {
        out << data[i].time << "," << best_fit[i] + noise * Subs::gauss2(seed) << "";
    }
    out.close();

    Helpers::plot_model(data, best_fit, false, copy, config["plot_device"].get<string>());

    Helpers::write_config_and_model_to_json(model, config, config["output_file_path"].get<string>()+".json");

    return 0;
}
