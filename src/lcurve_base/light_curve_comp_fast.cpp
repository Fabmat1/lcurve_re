#include "lcurve.h"
#include "constants.h"
#include <algorithm>
#include <iostream>

void Lcurve::light_curve_comp_fast(const Lcurve::Model &mdl,
                                   const Lcurve::Data &data, bool scale,
                                   bool rdata, bool info,
                                   std::vector<double> &sfac,
                                   std::vector<double> &calc, double &wdwarf,
                                   double &chisq, double &wnok,
                                   double &logg1, double &logg2, 
                                   double &rv1, double &rv2,
                                   int max_model_points) {
    
    // Determine if we need to subsample
    size_t n_data = data.size();
    bool need_interpolation = n_data > max_model_points;
    
    if (!need_interpolation) {
        // Use standard computation if data points are few
        light_curve_comp(mdl, data, scale, rdata, info, sfac, calc, 
                        wdwarf, chisq, wnok, logg1, logg2, rv1, rv2);
        return;
    }

    // Create subsampled data for model calculation
    Lcurve::Data model_data;
    std::vector<size_t> sample_indices;
    
    // Uniform sampling with endpoints included
    for (int i = 0; i < max_model_points; ++i) {
        size_t idx = i * (n_data - 1) / (max_model_points - 1);
        model_data.push_back(data[idx]);
        sample_indices.push_back(idx);
    }
    
    // Compute model at sample points
    std::vector<double> model_calc;
    light_curve_comp(mdl, model_data, scale, rdata, info, sfac, model_calc,
                    wdwarf, chisq, wnok, logg1, logg2, rv1, rv2);
    
    // Interpolate to all data points
    calc.resize(n_data);
    
    #pragma omp parallel for
    for (size_t i = 0; i < n_data; ++i) {
        // Find surrounding sample points
        auto it = std::lower_bound(sample_indices.begin(), sample_indices.end(), i);
        
        if (it == sample_indices.begin()) {
            // Before first sample point
            calc[i] = model_calc[0];
        } else if (it == sample_indices.end()) {
            // After last sample point
            calc[i] = model_calc.back();
        } else {
            // Linear interpolation between surrounding points
            size_t idx2 = std::distance(sample_indices.begin(), it);
            size_t idx1 = idx2 - 1;
            size_t i1 = sample_indices[idx1];
            size_t i2 = sample_indices[idx2];
            
            double t = (data[i].time - data[i1].time) / 
                      (data[i2].time - data[i1].time);
            calc[i] = model_calc[idx1] * (1.0 - t) + model_calc[idx2] * t;
        }
    }
    
    // Recalculate chi-squared with interpolated values if needed
    if (rdata && need_interpolation) {
        wnok = 0.;
        chisq = 0.;
        for (size_t i = 0; i < n_data; ++i) {
            if (data[i].weight > 0.) {
                wnok += data[i].weight;
                chisq += data[i].weight * Subs::sqr((data[i].flux - calc[i]) / data[i].ferr);
            }
        }
    }
}