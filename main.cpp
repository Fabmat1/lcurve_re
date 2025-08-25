// Minimal Modern re-adaptation of Tom Marsh's lcurve
// Makes the lightcurve modelling standalone and simplifies installation
// Features additional improvements such as a fixed stellar radius relation during fits
// This is held together with duct tape and good intentions and solely
// exists cause the lcurve install is broken as of 2025

#include <iostream>
#include <string>
#include <cmath>
#include <nlohmann/json.hpp>
#include "src/lcurve_base/lcurve.h"
#include "src/new_helpers.h"
#include "src/new_subs.h"

using namespace std;
using json = nlohmann::json;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <config_file.json>" << endl;
        return 1;
    }

    // Read configuration
    string config_file = argv[1];

    auto model_config_pair = Helpers::load_model_and_config_from_json(config_file);

    Lcurve::Model model = model_config_pair.first;
    json config = model_config_pair.second;

    auto data_copy_pair = Helpers::read_and_copy_lightcurve_from_file(config["data_file_path"]);

    Lcurve::Data data = data_copy_pair.first;
    Lcurve::Data copy = data_copy_pair.second;
    bool no_file = data.empty();

    double noise = 0.0;
    if (no_file) {
        auto fake_data = Helpers::generate_fake_data(config["fake_data_file_path"]);
        data = fake_data;
    }
    else {
        noise = config["noise"].get<double>();
    }

    int seed; bool scale; vector<double> sfac;
    Helpers::load_seed_scale_sfac(config, no_file, model, seed, scale, sfac);

    vector<double> fit;
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


    string sout = config["output_file_path"].get<string>();
    string device = config["plot_device"].get<string>();
    if(device!="none" && device!="null"){
        Helpers::plot_model(data, fit, no_file, copy, device);
    }

    for (long unsigned int i = 0; i < data.size(); i++) {
        data[i].flux = fit[i] + noise*Subs::gauss2(seed);
    }

    Helpers::write_data(data, sout);
    return 0;
}
