// Various helper functions and classes,
// added to lcurve by me to make coding with
// lcurve a better and easier experience

#ifndef NEW_HELPERS_H
#define NEW_HELPERS_H

#include "lcurve_base/lcurve.h"
#include "gnuplot-iostream.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;
using namespace std;

namespace Helpers {
    void plot_model(Lcurve::Data data, Subs::Array1D<double> fit, bool no_file, Lcurve::Data copy, const string &device) {
        // 1) X‐bounds + centering
        double x1 = data[0].time, x2 = data[0].time;
        for (size_t i = 1; i < data.size(); ++i) {
            x1 = min(x1, data[i].time);
            x2 = max(x2, data[i].time);
        }
        double con = 0.0;
        if (x2 - x1 < 0.01 * abs((x1 + x2) / 2.0)) {
            con = x1;
            x1 -= con;
            x2 -= con;
        }
        double xr = x2 - x1;
        x1 -= xr / 10.0;
        x2 += xr / 10.0;

        // 2) Build datasets, track flux‐ and residual‐ranges
        vector<pair<double, double>> model_line, obs_pts, resid_pts;
        vector<tuple<double, double, double>> obs_err, resid_err;

        double fy1 = 0, fy2 = 0, r1 = 0, r2 = 0;
        bool first_flux = true, first_resid = true;

        for (size_t i = 0; i < data.size(); ++i) {
            double t = data[i].time - con;
            double m = fit[i];               // model
            double d = data[i].flux;         // noisy data
            // Pull original σ from copy (saved before you scaled data)
            double s = (!no_file ? copy[i].ferr : data[i].ferr);
            if (s <= 0) s = 1e-3;              // tiny fallback, shouldn't happen

            // model line
            model_line.emplace_back(t, m);

            // data
            obs_err.emplace_back(t, d, s);
            obs_pts.emplace_back(t, d);

            // residual χ = (d – m)/σ
            double chi = (d - m) / s;
            resid_err.emplace_back(t, chi, 1.0);
            resid_pts.emplace_back(t, chi);

            // track flux‐range including error‐bars and model
            double fmin = min(d - s, m);
            double fmax = max(d + s, m);
            if (first_flux) {
                fy1 = fmin;
                fy2 = fmax;
                first_flux = false;
            } else {
                fy1 = min(fy1, fmin);
                fy2 = max(fy2, fmax);
            }

            // track residual‐range including ±1
            double rmin = chi - 1.0, rmax = chi + 1.0;
            if (first_resid) {
                r1 = rmin;
                r2 = rmax;
                first_resid = false;
            } else {
                r1 = min(r1, rmin);
                r2 = max(r2, rmax);
            }
        }

        // pad ranges by 10%
        double dflux = fy2 - fy1;
        if (dflux == 0) dflux = 1.0;
        fy1 -= 0.1 * dflux;
        fy2 += 0.1 * dflux;
        double dr = r2 - r1;
        if (dr == 0) dr = 1.0;
        r1 -= 0.1 * dr;
        r2 += 0.1 * dr;

        // 3) Launch gnuplot
        Gnuplot gp;
        if (device == "qt" || device == "wxt" || device == "x11")
            gp << "set terminal " << device << " persist\n";
        else
            gp << "set terminal qt persist\n";

        gp << "set multiplot layout 2,1 rowsfirst\n";

        // Top panel: Model line + Data errorbars + Data points
        gp << "set xlabel 'Time (phased)'\n"
           << "set ylabel 'Flux'\n"
           << "set xrange [" << x1 << ":" << x2 << "]\n"
           << "set yrange [" << fy1 << ":" << fy2 << "]\n"
           << "plot "
              "'-' with lines      lc rgb 'blue'  title 'Model', "
              "'-' with yerrorbars lc rgb 'red'   title 'Data err', "
              "'-' with points      lc rgb 'red'   pt 7    title 'Data'\n";
        gp.send1d(model_line);
        gp.send1d(obs_err);
        gp.send1d(obs_pts);

        // Bottom panel: Residuals χ ±1 + residual points + zero‐line
        gp << "set xlabel 'Time (phased)'\n"
           << "set ylabel 'Residual (χ)'\n"
           << "set xrange [" << x1 << ":" << x2 << "]\n"
           << "set yrange [" << r1 << ":" << r2 << "]\n"
           << "plot "
              "'-' with yerrorbars lc rgb 'purple' title 'χ ±1', "
              "'-' with points      lc rgb 'black'  pt 7    title 'χ', "
              "0 with lines         lc rgb 'gray'         title ''\n";
        gp.send1d(resid_err);
        gp.send1d(resid_pts);

        gp << "unset multiplot\n";
    }

    pair<Lcurve::Model, json> load_model_and_config_from_json(string config_file){
        ifstream config_stream(config_file);
        if (!config_stream) {
            throw runtime_error("Error: Unable to open configuration file: " + config_file);
        }
        json config;
        try { config_stream >> config; }
        catch (const json::parse_error &e) {
            throw runtime_error("JSON parse error: " + string(e.what()));
        }

        // Build model and data
        Lcurve::Model model(config);
        return make_pair(model, config);
    }

    pair<Lcurve::Data, Lcurve::Data> read_and_copy_lightcurve_from_file(string fpath){
        bool no_file = (Subs::toupper(fpath) == "NONE");
        Lcurve::Data data, copy;
        if (!no_file) {
            data.rasc(fpath);
            if (data.empty()) throw runtime_error("No data read from file.");
            copy = data;
        }
        else {
            cout << "'None' specified as data path, returning empty arrays!" << endl;
            return make_pair(data, copy);
        }
        return make_pair(data, copy);
    }

    Lcurve::Data generate_fake_data(json config){
        double time1 = 0, time2 = 0, expose = 0, noise = 0;
        int ntime = 0, ndivide = 0;

        time1 = config["time1"].get<double>();
        time2 = config["time2"].get<double>();
        ntime = config["ntime"].get<int>();
        expose = config["expose"].get<double>();
        ndivide = config["ndivide"].get<int>();
        noise = config["noise"].get<double>();
        Lcurve::Data fake_data(ntime);

        Lcurve::Datum datum = {0., expose, 0., noise, 1., ndivide};
        for (int i = 0; i < ntime; i++) {
            datum.time = time1 + (time2 - time1) * i / double(ntime - 1);
            fake_data[i] = datum;
        }
        return fake_data;
    }

    void write_data(Lcurve::Data data_out, string out_path){
        // note: noise will already have been added
        data_out.wrasc(out_path);
        cout << "Written data to " << out_path << endl;
    }

    void load_seed_scale_sfac(const json& config, bool no_file, const Lcurve::Model& model,
                              int32_t& seed, bool& scale, Subs::Buffer1D<double>& sfac) {
        // Load seed and ensure it is negative
        seed = config["seed"].get<int32_t>();
        if (seed > 0) seed = -seed;

        // Determine if scaling is enabled based on file and configuration
        scale = (!no_file && config["autoscale"].get<bool>());

        // Initialize the scale factors (sfac)
        sfac.resize(4);
        if (!scale) {
            if (model.iscale) {
                sfac[0] = config["sstar1"].get<double>();
                sfac[1] = config["sstar2"].get<double>();
                sfac[2] = config["sdisc"].get<double>();
                sfac[3] = config["sspot"].get<double>();
            } else {
                sfac[0] = config["ssfac"].get<double>();
            }
        }
    }

}


#endif //NEW_HELPERS_H
