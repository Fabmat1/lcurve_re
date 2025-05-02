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
    catch (const json::parse_error& e) {
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

    double time1=0, time2=0, expose=0, noise=0;
    int ntime=0, ndivide=0;
    if (no_file) {
        time1 = config["time1"].get<double>();
        time2 = config["time2"].get<double>();
        ntime  = config["ntime"].get<int>();
        expose = config["expose"].get<double>();
        ndivide= config["ndivide"].get<int>();
        noise  = config["noise"].get<double>();
    } else {
        noise  = config["noise"].get<double>();
    }

    if (no_file) {
        Lcurve::Datum datum = {0., expose, 0., noise, 1., ndivide};
        for (int i = 0; i < ntime; i++) {
            datum.time = time1 + (time2 - time1) * i / double(ntime - 1);
            data.push_back(datum);
        }
    }

    int32_t seed = config["seed"].get<int32_t>(); if (seed > 0) seed = -seed;
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

    if (!no_file) for (auto &d : data) d.ferr *= noise;

    if ((nfile == 0 || nfile == 1) && device != "none" && device != "null") {
        // Compute ranges
        double x1 = data[0].time, x2 = data[0].time;
        double y1 = data[0].flux - data[0].ferr;
        double y2 = data[0].flux + data[0].ferr;
        for (size_t i=1; i<data.size(); ++i) {
            x1 = min(x1, data[i].time);
            x2 = max(x2, data[i].time);
            y1 = min(y1, data[i].flux - data[i].ferr);
            y2 = max(y2, data[i].flux + data[i].ferr);
            if (!no_file) {
                y1 = min(y1, copy[i].flux - copy[i].ferr);
                y2 = max(y2, copy[i].flux + copy[i].ferr);
                y1 = min(y1, roff + copy[i].flux - data[i].flux - copy[i].ferr);
                y2 = max(y2, roff + copy[i].flux - data[i].flux + copy[i].ferr);
            }
        }
        double con = 0;
        if (x2-x1 < 0.01*fabs((x1+x2)/2.)) { con = x1; x1 -= con; x2 -= con; }
        double dx = x2-x1, dy = y2-y1;
        if (dx == 0) { x1 -= 1; x2 += 1; } else { x1 -= dx/10; x2 += dx/10; }
        if (dy == 0) { y1 -= 1; y2 += 1; } else { y1 -= dy/10; y2 += dy/10; }

        // Prepare datasets
        vector<tuple<double,double,double>> data_err;
        vector<pair<double,double>> data_line;
        for (auto &d: data) {
            double t = d.time - con;
            data_err.emplace_back(t, d.flux-d.ferr, d.flux+d.ferr);
            data_line.emplace_back(t, d.flux);
        }
        vector<tuple<double,double,double>> copy_err1, copy_err2;
        vector<pair<double,double>> copy_pts, copy_res;
        if (!no_file) {
            for (size_t i=0; i<copy.size(); ++i) {
                double t = copy[i].time - con;
                copy_err1.emplace_back(t, copy[i].flux-copy[i].ferr, copy[i].flux+copy[i].ferr);
                copy_pts.emplace_back(t, copy[i].flux);
                double res = roff + copy[i].flux - data[i].flux;
                copy_err2.emplace_back(t, res-copy[i].ferr, res+copy[i].ferr);
                copy_res.emplace_back(t, res);
            }
        }

        // Launch gnuplot
        Gnuplot gp;
        if (!sout.empty() && sout!="null") gp << "set output '" << sout << "'\n";
        gp << "set xrange ["<<x1<<":"<<x2<<"]\n"  \
           << "set yrange ["<<y1<<":"<<y2<<"]\n"  \
           << "set xlabel 'T - "<<con<<"'\n" \
           << "set ylabel ''\n"       \
           << "set style line 1 lt 1 lw 5 lc rgb 'black'\n" \
           << "set style line 2 lt 1 lw 1 lc rgb '#B3B3B3'\n" \
           << "set style line 3 lt 1 lw 3 lc rgb '#008000'\n";

        // Plot
        if (!no_file) {
            gp << "plot '-' using 1:2:3 with yerrorbars ls 2 title 'Copy err', " \
               <<     "'-' using 1:2 with points ls 3 title 'Copy', " \
               <<     "'-' using 1:2:3 with yerrorbars ls 2 notitle, " \
               <<     "'-' using 1:2 with points ls 3 notitle, " \
               <<     "'-' using 1:2 with lines ls 1 title 'Data'\n";
            gp.send1d(copy_err1);
            gp.send1d(copy_pts);
            gp.send1d(copy_err2);
            gp.send1d(copy_res);
            gp.send1d(data_line);
        } else {
            if (noise==0.0) {
                gp << "plot '-' using 1:2 with lines ls 1 title 'Data'\n";
                gp.send1d(data_line);
            } else {
                gp << "plot '-' using 1:2:3 with yerrorbars ls 2 title 'Data err', " \
                   << "'-' using 1:2 with points ls 3 title 'Data pt'\n";
                gp.send1d(data_err);
                gp.send1d(data_line);
            }
        }
    }

    return 0;
}
