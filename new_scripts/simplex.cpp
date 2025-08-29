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
//----------------------------------------------------------

std::vector<double> nelder_mead(
          const std::function<double(const std::vector<double>&)>& f,
          const std::vector<double>& x0,
          const std::vector<double>& step,
          const std::vector<std::pair<double,double>>& limits,
          int    max_iter = 5'000,
          double ftol     = 1.e-5)          //  <-- ftol = NR “ftol”
{
    const int n = static_cast<int>(x0.size());
    const int m = n + 1;                    // simplex size

    // -----------------------------------------------------------------
    // Build initial simplex:  x0  +  step[i] ê_i   (clipped to limits)
    // -----------------------------------------------------------------
    std::vector<std::vector<double>> simplex(m, x0);
    for(int i = 0; i < n; ++i){
        simplex[i+1][i] = std::clamp(x0[i] + step[i], limits[i].first,
                                                 limits[i].second);
    }

    // Evaluate function at the vertices
    std::vector<double> fval(m);
    for(int i = 0; i < m; ++i) fval[i] = f(simplex[i]);

    // Nelder-Mead loop
    const double α = 1.0;   // reflection
    const double γ = 2.0;   // expansion
    const double ρ = 0.5;   // contraction
    const double σ = 0.5;   // shrink

    for(int iter = 0; iter < max_iter; ++iter){

        // Rank vertices
        std::vector<int> idx(m);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b){ return fval[a] < fval[b]; });

        // -----------------------------------------------------------------
        //  Convergence test  (Numerical-Recipes style, relative)
        // -----------------------------------------------------------------
        double rtol = 2.0*std::fabs(fval[idx[m-1]] - fval[idx[0]]) /
                      (std::fabs(fval[idx[m-1]]) + std::fabs(fval[idx[0]]) + 1e-30);
        if(rtol < ftol) break;

        const std::vector<double>& x_best  = simplex[idx[0]];
        const std::vector<double>& x_worst = simplex[idx[m-1]];

        // Centroid of all but worst
        std::vector<double> x_cent(n,0.0);
        for(int k = 0; k < m-1; ++k){
            const std::vector<double>& v = simplex[idx[k]];
            for(int j = 0; j < n; ++j) x_cent[j] += v[j];
        }
        for(double& v : x_cent) v /= n;

        // -------- reflection ------------------------------------------------
        std::vector<double> x_ref(n);
        for(int j = 0; j < n; ++j){
            x_ref[j] = std::clamp(x_cent[j] + α*(x_cent[j] - x_worst[j]),
                             limits[j].first, limits[j].second);
        }
        double f_ref = f(x_ref);

        if(f_ref < fval[idx[0]]){

            // -------- expansion ---------------------------------------------
            std::vector<double> x_exp(n);
            for(int j = 0; j < n; ++j){
                x_exp[j] = std::clamp(x_cent[j] + γ*(x_ref[j] - x_cent[j]),
                                 limits[j].first, limits[j].second);
            }
            double f_exp = f(x_exp);

            simplex[idx[m-1]] = (f_exp < f_ref) ? x_exp : x_ref;
            fval   [idx[m-1]] = std::min(f_ref, f_exp);

        }else if(f_ref < fval[idx[m-2]]){

            // accept reflected point
            simplex[idx[m-1]] = x_ref;
            fval   [idx[m-1]] = f_ref;

        }else{

            // -------- contraction -------------------------------------------
            std::vector<double> x_con(n);
            for(int j = 0; j < n; ++j){
                x_con[j] = std::clamp(x_cent[j] + ρ*(x_worst[j] - x_cent[j]),
                                 limits[j].first, limits[j].second);
            }
            double f_con = f(x_con);

            if(f_con < fval[idx[m-1]]){
                simplex[idx[m-1]] = x_con;
                fval   [idx[m-1]] = f_con;
            }else{

                // -------- shrink (FIXED) -------------------------------------
                for(int k = 1; k < m; ++k){
                    int i = idx[k];                // real vertex to move
                    for(int j = 0; j < n; ++j){
                        simplex[i][j] = std::clamp(x_best[j] +
                                              σ*(simplex[i][j] - x_best[j]),
                                              limits[j].first, limits[j].second);
                    }
                    fval[i] = f(simplex[i]);
                }
            }
        }

        if(iter % 10 == 0){
            std::cout << "Iter " << iter
                      << "  χ²_best = " << fval[idx[0]] << std::endl;
        }
    }

    int best = std::min_element(fval.begin(), fval.end()) - fval.begin();
    return simplex[best];
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
    std::vector<double> x0(npar), steps(npar);        // <-- add this line
    Subs::Array1D<double> init_pars = model.get_param();
    Subs::Array1D<double> ranges = model.get_range();
    auto limits = model.get_limit();
    for(int i = 0; i < npar; ++i){
        x0[i]    = init_pars[i];
        steps[i] = ranges[i];          //  << was dsteps[i]
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
                                      config.value("simplex_tol", 1e-5));
    // Ensure returned vector matches parameter count
    if(best.size() != x0.size()) {
        best.resize(x0.size());
    }
    double best_chi = obj(best);
    auto t1 = Clock::now();


    // Report
    cout << "Simplex optimization completed in "
         << chrono::duration<double>(t1 - t0).count() << "s" << endl;

    int    dof            = static_cast<int>(data.size()) - npar;
    double red_chi        = best_chi / std::max(1, dof);

    std::cout << "Best χ²        = " << best_chi  << '\n'
              << "Reduced χ²     = " << red_chi   << '\n';
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
