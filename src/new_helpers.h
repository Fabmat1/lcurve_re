// Various helper functions and classes,
// added to lcurve by me to make coding with
// lcurve a better and easier experience

#ifndef NEW_HELPERS_H
#define NEW_HELPERS_H

#include "lcurve_base/lcurve.h"
#include "gnuplot-iostream.h"
#include "model.h"       // <-- gives access to struct Pparam

#include "lcurve_base/constants.h"
#include <nlohmann/json.hpp>
using json = nlohmann::json;
using namespace std;

namespace {

/* Convert a Pparam to the same single-line representation that the
   parser (Pparam::Pparam(const string&)) understands.            */
std::string pparam_to_string(const Lcurve::Pparam &p) {
    std::ostringstream oss;
    oss << std::setprecision(10) << std::fixed << p.value << " "
        << std::setprecision(6)  << std::fixed << p.range << " "
        << std::setprecision(6)  << std::fixed << p.dstep << " "
        << int(p.vary) << " " << int(p.defined);
    return oss.str();
}

}   // unnamed namespace

namespace Helpers {
    inline void plot_model(Lcurve::Data data,
                    vector<double> fit,
                    bool no_file,
                    Lcurve::Data copy,
                    const string &device) {
        // 1) Compute x‐bounds and centering
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

        // 2) Build datasets and track ranges
        vector<pair<double, double> > model_line, obs_pts, resid_pts;
        vector<tuple<double, double, double> > obs_err, resid_err;

        double fy1 = 0, fy2 = 0, r1 = 0, r2 = 0;
        bool first_flux = true, first_resid = true;

        for (size_t i = 0; i < data.size(); ++i) {
            double t = data[i].time - con;
            double m = fit[i];
            double d = data[i].flux;
            double s = (!no_file ? copy[i].ferr : data[i].ferr);
            if (s <= 0) s = 1e-3;

            model_line.emplace_back(t, m);
            obs_err.emplace_back(t, d, s);
            obs_pts.emplace_back(t, d);

            double chi = (d - m) / s;
            resid_err.emplace_back(t, chi, 1.0);
            resid_pts.emplace_back(t, chi);

            // flux range
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
            // residual range
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

        // --- send named datablocks ---
        gp << "$Model << EOD\n";
        gp.send1d(model_line);
        gp << "EOD\n";
        gp << "$ObsErr << EOD\n";
        gp.send1d(obs_err);
        gp << "EOD\n";
        gp << "$ObsPts << EOD\n";
        gp.send1d(obs_pts);
        gp << "EOD\n";
        gp << "$ResErr << EOD\n";
        gp.send1d(resid_err);
        gp << "EOD\n";
        gp << "$ResPts << EOD\n";
        gp.send1d(resid_pts);
        gp << "EOD\n";

        // Top panel: Flux
        gp << "set xlabel 'Time (phased)'\n"
                << "set ylabel 'Flux'\n"
                << "set xrange [" << x1 << ":" << x2 << "]\n"
                << "set yrange [" << fy1 << ":" << fy2 << "]\n"
                << "plot "
                "$ObsErr   with yerrorbars lc rgb 'black'   title 'Data err', "
                "$ObsPts   with points      lc rgb 'black'   pt 7    title 'Data', "
                "$Model    with lines      lc rgb 'red'  title 'Model'\n";


        // Bottom panel: Residuals
        gp << "set xlabel 'Time (phased)'\n"
                << "set ylabel 'Residual (χ)'\n"
                << "set xrange [" << x1 << ":" << x2 << "]\n"
                << "set yrange [" << r1 << ":" << r2 << "]\n"
                << "plot "
                "$ResErr   with yerrorbars lc rgb 'purple' title 'χ ±1', "
                "$ResPts   with points      lc rgb 'black'  pt 7    title 'χ', "
                "0         with lines         lc rgb 'gray'         title ''\n";

        gp << "unset multiplot\n";
    }


    inline void plot_model_live(Lcurve::Data data,
                         vector<double> fit,
                         bool no_file,
                         Lcurve::Data copy,
                         Gnuplot &gp) {
        // 1) Compute x‐bounds and centering
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

        // 2) Build datasets and track ranges
        vector<pair<double, double> > model_line, obs_pts, resid_pts;
        vector<tuple<double, double, double> > obs_err, resid_err;
        double fy1 = 0, fy2 = 0, r1 = 0, r2 = 0;
        bool first_flux = true, first_resid = true;

        for (size_t i = 0; i < data.size(); ++i) {
            double t = data[i].time - con;
            double m = fit[i];
            double d = data[i].flux;
            double s = (!no_file ? copy[i].ferr : data[i].ferr);
            if (s <= 0) s = 1e-3;

            model_line.emplace_back(t, m);
            obs_err.emplace_back(t, d, s);
            obs_pts.emplace_back(t, d);

            double chi = (d - m) / s;
            resid_err.emplace_back(t, chi, 1.0);
            resid_pts.emplace_back(t, chi);

            double fmin = min(d - s, m), fmax = max(d + s, m);
            if (first_flux) {
                fy1 = fmin;
                fy2 = fmax;
                first_flux = false;
            } else {
                fy1 = min(fy1, fmin);
                fy2 = max(fy2, fmax);
            }

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
        double dflux = fy2 - fy1;
        if (dflux == 0) dflux = 1.0;
        fy1 -= 0.1 * dflux;
        fy2 += 0.1 * dflux;
        double dr = r2 - r1;
        if (dr == 0) dr = 1.0;
        r1 -= 0.1 * dr;
        r2 += 0.1 * dr;

        // 3) Redraw in existing gp
        gp << "clear\n"
                << "set multiplot layout 2,1 rowsfirst\n";

        // send blocks
        gp << "$Model << EOD\n";
        gp.send1d(model_line);
        gp << "EOD\n";
        gp << "$ObsErr << EOD\n";
        gp.send1d(obs_err);
        gp << "EOD\n";
        gp << "$ObsPts << EOD\n";
        gp.send1d(obs_pts);
        gp << "EOD\n";
        gp << "$ResErr << EOD\n";
        gp.send1d(resid_err);
        gp << "EOD\n";
        gp << "$ResPts << EOD\n";
        gp.send1d(resid_pts);
        gp << "EOD\n";

        // top
        gp << "set xlabel 'Time (phased)'\n"
                << "set ylabel 'Flux'\n"
                << "set xrange [" << x1 << ":" << x2 << "]\n"
                << "set yrange [" << fy1 << ":" << fy2 << "]\n"
                << "plot "
                "$ObsErr with yerrorbars lc rgb 'black' title 'Data err', "
                "$ObsPts with points lc rgb 'black' pt 7 title 'Data', "
                "$Model with lines lc rgb 'red' title 'Model'\n";


        // bottom
        gp << "set xlabel 'Time (phased)'\n"
                << "set ylabel 'Residual (χ)'\n"
                << "set xrange [" << x1 << ":" << x2 << "]\n"
                << "set yrange [" << r1 << ":" << r2 << "]\n"
                << "plot $ResErr with yerrorbars lc rgb 'purple' title 'χ ±1', "
                "$ResPts with points lc rgb 'black' pt 7 title 'χ', "
                "0 with lines lc rgb 'gray'\n";

        gp << "unset multiplot\n"
                << flush;
    }


    inline pair<Lcurve::Model, json> load_model_and_config_from_json(string config_file) {
        ifstream config_stream(config_file);
        if (!config_stream) {
            throw runtime_error("Error: Unable to open configuration file: " + config_file);
        }
        json config;
        try { config_stream >> config; } catch (const json::parse_error &e) {
            throw runtime_error("JSON parse error: " + string(e.what()));
        }

        // Build model and data
        Lcurve::Model model(config);
        return make_pair(model, config);
    }

    inline pair<Lcurve::Data, Lcurve::Data> read_and_copy_lightcurve_from_file(string fpath) {
        bool no_file = (Subs::toupper(fpath) == "NONE");
        Lcurve::Data data, copy;
        if (!no_file) {
            data.rasc(fpath);
            if (data.empty()) throw runtime_error("No data read from file.");
            copy = data;
        } else {
            cout << "'None' specified as data path, returning empty arrays!" << endl;
            return make_pair(data, copy);
        }
        return make_pair(data, copy);
    }

    inline Lcurve::Data generate_fake_data(json config) {
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

    inline void write_data(Lcurve::Data data_out, string out_path) {
        // note: noise will already have been added
        data_out.wrasc(out_path);
        cout << "Written data to " << out_path << endl;
    }

    inline void load_seed_scale_sfac(const json &config, bool no_file, const Lcurve::Model &model,
                              int32_t &seed, bool &scale, vector<double> &sfac) {
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

    inline double velocity_scale_from_inclination(double inclination, double rv_obs, double mass_ratio) {
        return rv_obs / sin(inclination * M_PI / 180.0) * (mass_ratio + 1) / mass_ratio;
    }

    // Compute orbital separation in solar radii
    inline double compute_scaled_r1(double r1, double velocity_scale, double P_days) {
        // Convert to SI units
        double P = P_days * Constants::IDAY;

        // Convert to solar radii
        return r1/(1000*velocity_scale*P / (Constants::RSUN*2*M_PI));
    }

    inline std::tuple<double, double, double> parseThreeDoubles(const std::string& input) {
        std::istringstream iss(input);
        double a, b, c;
        if (!(iss >> a >> b >> c)) {
            throw std::runtime_error("Input string does not contain exactly three double values.");
        }
        // Ensure no extra input
        std::string leftover;
        if (iss >> leftover) {
            throw std::runtime_error("Input string contains more than three values.");
        }
        return std::make_tuple(a, b, c);
    }


    /*  Write a copy of the configuration, but with the
        model_parameters updated to the values held in ‘model’.     */
    inline void write_config_and_model_to_json(const Lcurve::Model &model,
                                             json                 config,
                                             const std::string   &out_path)
    {
        if (!config.contains("model_parameters")
            || !config["model_parameters"].is_object())
            throw std::runtime_error(
                 "write_config_and_model_to_json: config misses "
                 "'model_parameters' object");

        auto &mp = config["model_parameters"];

    #define UPDATE(name)  mp[#name] = pparam_to_string(model.name)

        /* -------- fundamental & geometric -------- */
        UPDATE(q);              UPDATE(iangle);
        UPDATE(r1);             UPDATE(r2);
        UPDATE(cphi3);          UPDATE(cphi4);
        UPDATE(spin1);          UPDATE(spin2);
        UPDATE(t1);             UPDATE(t2);

        /* -------- limb darkening -------- */
        UPDATE(ldc1_1); UPDATE(ldc1_2); UPDATE(ldc1_3); UPDATE(ldc1_4);
        UPDATE(ldc2_1); UPDATE(ldc2_2); UPDATE(ldc2_3); UPDATE(ldc2_4);

        /* -------- beaming / velocity scale -------- */
        UPDATE(velocity_scale); UPDATE(beam_factor1); UPDATE(beam_factor2);

        /* -------- ephemeris & timing -------- */
        UPDATE(t0);  UPDATE(period);  UPDATE(pdot);  UPDATE(deltat);

        /* -------- miscellaneous coefficients -------- */
        UPDATE(gravity_dark1);  UPDATE(gravity_dark2);
        UPDATE(absorb);         UPDATE(slope);
        UPDATE(quad);           UPDATE(cube);        UPDATE(third);

        /* -------- accretion disc -------- */
        UPDATE(rdisc1);         UPDATE(rdisc2);
        UPDATE(height_disc);    UPDATE(beta_disc);
        UPDATE(temp_disc);      UPDATE(texp_disc);
        UPDATE(lin_limb_disc);  UPDATE(quad_limb_disc);
        UPDATE(temp_edge);      UPDATE(absorb_edge);

        /* -------- bright spot on disc -------- */
        UPDATE(radius_spot); UPDATE(length_spot); UPDATE(height_spot);
        UPDATE(expon_spot);  UPDATE(epow_spot);
        UPDATE(angle_spot);  UPDATE(yaw_spot);    UPDATE(temp_spot);
        UPDATE(tilt_spot);   UPDATE(cfrac_spot);

        /* -------- star-spots (up to 3 per star) -------- */
        UPDATE(stsp11_long); UPDATE(stsp11_lat); UPDATE(stsp11_fwhm); UPDATE(stsp11_tcen);
        UPDATE(stsp12_long); UPDATE(stsp12_lat); UPDATE(stsp12_fwhm); UPDATE(stsp12_tcen);
        UPDATE(stsp13_long); UPDATE(stsp13_lat); UPDATE(stsp13_fwhm); UPDATE(stsp13_tcen);
        UPDATE(stsp21_long); UPDATE(stsp21_lat); UPDATE(stsp21_fwhm); UPDATE(stsp21_tcen);
        UPDATE(stsp22_long); UPDATE(stsp22_lat); UPDATE(stsp22_fwhm); UPDATE(stsp22_tcen);

        /* -------- uniform equatorial spot -------- */
        UPDATE(uesp_long1); UPDATE(uesp_long2);
        UPDATE(uesp_lathw); UPDATE(uesp_taper); UPDATE(uesp_temp);

    #undef UPDATE

        /* pretty-print (2-space indent) */
        std::ofstream fout(out_path);
        if (!fout)
            throw std::runtime_error("Cannot open '" + out_path + "' for writing");
        fout << std::setw(2) << config << '\n';
    }
}


#endif //NEW_HELPERS_H
