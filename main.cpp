// Minimal Modern re-adaptation of Tom Marsh's lcurve
// Makes the lightcurve modelling standalone and simplifies installation
// Features additional improvements such as a fixed stellar radius relation during fits
// This is held together with duct tape and good intentions and solely
// exists cause the lcurve install is broken as of 2025

#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <string>
#include <utility>
#include <cmath>
#include <nlohmann/json.hpp>
#include "gnuplot-iostream.h"
#include "src/lcurve_base/lcurve.h"

using namespace std;
using json = nlohmann::json;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <config_file.json>" << endl;
        return 1;
    }

    // Read configuration
    string config_file = argv[1];
    ifstream config_stream(config_file);
    if (!config_stream) {
        cerr << "Error: Unable to open configuration file: " << config_file << endl;
        return 1;
    }
    json config;
    try { config_stream >> config; }
    catch (const json::parse_error &e) {
        cerr << "JSON parse error: " << e.what() << endl;
        return 1;
    }

    // Build model and data
    Lcurve::Model model(config);
    string sdata = config["data_file_path"];
    bool no_file = (Subs::toupper(sdata) == "NONE");
    Lcurve::Data data, copy;
    if (!no_file) {
        data.rasc(sdata);
        if (data.empty()) throw runtime_error("No data read from file.");
        copy = data;
    }

    double time1 = 0, time2 = 0, expose = 0, noise = 0;
    int ntime = 0, ndivide = 0;
    if (no_file) {
        time1 = config["time1"].get<double>();
        time2 = config["time2"].get<double>();
        ntime = config["ntime"].get<int>();
        expose = config["expose"].get<double>();
        ndivide = config["ndivide"].get<int>();
        noise = config["noise"].get<double>();
    } else {
        noise = config["noise"].get<double>();
    }

    if (no_file) {
        Lcurve::Datum datum = {0., expose, 0., noise, 1., ndivide};
        for (int i = 0; i < ntime; i++) {
            datum.time = time1 + (time2 - time1) * i / double(ntime - 1);
            data.push_back(datum);
        }
    }

    int32_t seed = config["seed"].get<int32_t>();
    if (seed > 0) seed = -seed;
    int nfile = config["nfile"].get<int>();
    string sout = config["output_file_path"].get<string>();
    string device = config["plot_device"].get<string>();
    double roff = config["residual_offset"].get<double>();
    bool scale = (!no_file && config["autoscale"].get<bool>());

    Subs::Buffer1D<double> sfac(4);
    if (!scale) {
        if (model.iscale) {
            sfac[0] = config["sstar1"].get<double>();
            sfac[1] = config["sstar2"].get<double>();
            sfac[2] = config["sdisc"].get<double>();
            sfac[3] = config["sspot"].get<double>();
        } else sfac[0] = config["ssfac"].get<double>();
    }

    Subs::Array1D<double> fit;
    double wdwarf, chisq, wnok, logg1, logg2, rv1, rv2;
    Lcurve::light_curve_comp(model, data, scale, !no_file, true, sfac,
                             fit, wdwarf, chisq, wnok,
                             logg1, logg2, rv1, rv2);

    if (!no_file) {
        cout << "Weighted chi**2 = " << chisq << ", wnok = " << wnok << endl;
        if (model.iscale)
            cout << "Scale factors = " << sfac[0] << ", " << sfac[1]
                 << ", " << sfac[2] << ", " << sfac[3] << endl;
        else
            cout << "Scale factor = " << sfac[0] << endl;
    }
    cout << "White dwarf's contribution = " << wdwarf << endl;
    cout << "log10(g1 [cgs]) = " << logg1 << endl;
    cout << "log10(g2 [cgs]) = " << logg2 << endl;
    cout << "Vol-averaged r1 = " << rv1 << endl;
    cout << "Vol-averaged r2 = " << rv2 << endl;

    if (!no_file) for (auto &d: data) d.ferr *= noise;

    if((nfile == 0 || nfile == 1) && device!="none" && device!="null"){
        // 1) X‐bounds + centering
        double x1 = data[0].time, x2 = data[0].time;
        for(size_t i=1; i<data.size(); ++i){
            x1 = std::min(x1, data[i].time);
            x2 = std::max(x2, data[i].time);
        }
        double con = 0.0;
        if(x2 - x1 < 0.01*std::abs((x1 + x2)/2.0)){
            con = x1;  x1 -= con;  x2 -= con;
        }
        double xr = x2 - x1;  x1 -= xr/10.0;  x2 += xr/10.0;

        // 2) Build datasets, track flux‐ and residual‐ranges
        std::vector<std::pair<double,double>>        model_line, obs_pts, resid_pts;
        std::vector<std::tuple<double,double,double>> obs_err, resid_err;

        double fy1=0, fy2=0, r1=0, r2=0;
        bool first_flux=true, first_resid=true;

        for(size_t i=0; i<data.size(); ++i){
            double t = data[i].time - con;
            double m = fit[i];               // model
            double d = data[i].flux;         // noisy data
            // Pull original σ from copy (saved before you scaled data)
            double s = (!no_file ? copy[i].ferr : data[i].ferr);
            if(s <= 0) s = 1e-3;              // tiny fallback, shouldn't happen

            // model line
            model_line.emplace_back(t, m);

            // data
            obs_err.emplace_back(t, d, s);
            obs_pts.emplace_back(t, d);

            // residual χ = (d – m)/σ
            double chi = (d - m)/s;
            resid_err.emplace_back(t, chi, 1.0);
            resid_pts.emplace_back(t, chi);

            // track flux‐range including error‐bars and model
            double fmin = std::min(d - s, m);
            double fmax = std::max(d + s, m);
            if(first_flux){
                fy1 = fmin;  fy2 = fmax;  first_flux=false;
            } else {
                fy1 = std::min(fy1, fmin);
                fy2 = std::max(fy2, fmax);
            }

            // track residual‐range including ±1
            double rmin = chi - 1.0, rmax = chi + 1.0;
            if(first_resid){
                r1 = rmin;  r2 = rmax;  first_resid=false;
            } else {
                r1 = std::min(r1, rmin);
                r2 = std::max(r2, rmax);
            }
        }

        // pad ranges by 10%
        double dflux = fy2 - fy1;  if(dflux==0) dflux=1.0;
        fy1 -= 0.1*dflux;  fy2 += 0.1*dflux;
        double dr    = r2 - r1;    if(dr==0) dr=1.0;
        r1  -= 0.1*dr;    r2  += 0.1*dr;

        // 3) Launch gnuplot
        Gnuplot gp;
        if(device=="qt"||device=="wxt"||device=="x11")
            gp << "set terminal " << device << " persist\n";
        else
            gp << "set terminal qt persist\n";

        gp << "set multiplot layout 2,1 rowsfirst\n";

        // Top panel: Model line + Data errorbars + Data points
        gp << "set xlabel 'Time (phased)'\n"
           << "set ylabel 'Flux'\n"
           << "set xrange ["<<x1<<":"<<x2<<"]\n"
           << "set yrange ["<<fy1<<":"<<fy2<<"]\n"
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
           << "set xrange ["<<x1<<":"<<x2<<"]\n"
           << "set yrange ["<<r1<<":"<<r2<<"]\n"
           << "plot "
              "'-' with yerrorbars lc rgb 'purple' title 'χ ±1', "
              "'-' with points      lc rgb 'black'  pt 7    title 'χ', "
              "0 with lines         lc rgb 'gray'         title ''\n";
        gp.send1d(resid_err);
        gp.send1d(resid_pts);

        gp << "unset multiplot\n";
    }
    return 0;
}
