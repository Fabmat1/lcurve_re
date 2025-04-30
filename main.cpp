// Minimal Modern re-adaptation of Tom Marsh's lcurve
// Makes the lightcurve modelling standalone and simplifies installation
// Features additional improvements such as a fixed stellar radius relation during fits
// This is held together with duct tape and good intentions and solely
// exists cause the lcurve install is broken as of 2025

#include <iostream>
#include <nlohmann/json.hpp>
#include <fstream>

#include "src/lcurve.h"
#include "src/model.h"

using namespace std;
using json = nlohmann::json;

// Main function that functions similarly
int main(int argc, char* argv[]) {
    // The prompts and unintuitive caching of inputs from lroche is replaced by
    // a simple .json configuration file that can be easily swapped out or in
    // for ease of specifying different parameters

    // Check if configuration file is provided via CLI
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <config_file.json>" << endl;
        return 1;
    }

    string config_file = argv[1];

    // Open and parse the JSON configuration file
    ifstream config_stream(config_file);
    if (!config_stream.is_open()) {
        cerr << "Error: Unable to open configuration file: " << config_file << endl;
        return 1;
    }

    json config;
    try {
        config_stream >> config;
    } catch (const json::parse_error& e) {
        cerr << "JSON parse error: " << e.what() << endl;
        return 1;
    }

    Model model(config);

    // From here we follow the parsing like lroche also does
    // There is no need to
    // First checking if a data file was provided

    string sdata = config["data_file_path"];
    bool no_file = (Subs::toupper(sdata) == "NONE");
    Lcurve::Data data, copy;
    if(!no_file){
        data.rasc(sdata);
        if(data.empty())
            throw runtime_error("No data read from file.");
        copy = data;
    }

    // If there is no data we need to define a grid

    double time1, time2, expose, noise;
    int ntime, ndivide;
    if(no_file){
        time1 = config["time1"].get<double>();
        time2 = config["time2"].get<double>();
        ntime = config["ntime"].get<int>();
        expose = config["expose"].get<double>();
        ndivide = config["ndivide"].get<double>();
        noise = config["noise"].get<double>();
    }else{
        noise = config["noise"].get<double>();
    }

    if(no_file){
        // Build fake data
        Lcurve::Datum datum = {0., expose, 0., noise, 1., ndivide};
        for(int i=0; i<ntime; i++){
            datum.time   = time1 + (time2-time1)*i/(ntime-1);
            data.push_back(datum);
        }
    }

    // Definitions following the pattern from lroche.cc from Tom Marsh.
    // Subs::INT4 is replaced by int32_t as they are the same thing (why??)
    int32_t seed;
    seed = config["seed"].get<int32_t>();
    if(seed > 0) seed = -seed;
    int nfile = config["nfile"].get<int>();
    string sout = config["output_file_path"].get<string>();
    string device = config["plot_device"].get<string>();
    double roff = config["residual_offset"].get<double>();
    bool scale = false;

    if (!no_file) {
        scale = config["scale"].get<bool>();
    }
    Subs::Buffer1D<double> sfac(4);

    if (!scale) {
        if (model.iscale) {
            sfac[0] = config["sstar1"].get<double>();
            sfac[1] = config["sstar2"].get<double>();
            sfac[2] = config["sdisc"].get<double>();
            sfac[3] = config["sspot"].get<double>();
        }
        else {
            sfac[0] = config["ssfac"].get<double>();
        }
    }

    Subs::Array1D<double> fit;
    double wdwarf, chisq, wnok, logg1, logg2, rv1, rv2;
    Subs::Format form(12);
    Lcurve::light_curve_comp(model, data, scale, !no_file, true, sfac,
                                     fit, wdwarf, chisq, wnok,
                                     logg1, logg2, rv1, rv2);

    return 0;
}
