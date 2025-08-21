#include "../src/new_helpers.h"
#include "../src/mass_ratio_pdf.h"
#include "gnuplot-iostream.h"
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <chrono>
#include <iomanip>


int main() {
    // ───────────────────────────────────────────────────────────────
    // 1) Initialize the grid with all parameters
    // ───────────────────────────────────────────────────────────────
    const double m1_mean = 0.82;
    const double m1_err = 0.17;
    const double m2_mean = 1.3;
    const double m2_err = 0.27;
    
    // New parameters for velocity and radius scales
    const double K_mean = 186.2;    // km/s
    const double K_err = 2.0;      // km/s
    const double R_mean = 0.309;     // solar radii
    const double R_err = 0.02;      // solar radii
    const double P_mean = 0.31955193;     // days
    const double P_err = 0.00000028;     // days
    
    const double test_inclination = 50.0;  // degrees
    const double q_eval = 1.57;
    const double vs_eval = 427.0;  // km/s
    const double rs_eval = 0.118;   // unitless
    
    std::cout << "=== Multi-Scale PDF Grid Test ===" << std::endl;
    std::cout << "Initializing grid with:" << std::endl;
    std::cout << "  m1 ~ N(" << m1_mean << ", " << m1_err << ") M_sun" << std::endl;
    std::cout << "  m2 ~ N(" << m2_mean << ", " << m2_err << ") M_sun" << std::endl;
    std::cout << "  K ~ N(" << K_mean << ", " << K_err << ") km/s" << std::endl;
    std::cout << "  R ~ N(" << R_mean << ", " << R_err << ") R_sun" << std::endl;
    std::cout << "  P ~ N(" << P_mean << ", " << P_err << ") days" << std::endl;
    std::cout << "  Test inclination: " << test_inclination << " degrees" << std::endl;
    
    auto start_init = std::chrono::high_resolution_clock::now();
    
    // Initialize grid with all parameters
    initialize_mass_ratio_pdf_grid(
        m1_mean, m1_err, m2_mean, m2_err,
        K_mean, K_err, R_mean, R_err, P_mean, P_err,
        15.0, 90.0,    // inclination range [15, 90] degrees
        500,            // inclination points
        500,           // q points
        500,           // v_s points
        500,           // r_s points
        100000000          // samples for accuracy
    );
    
    auto end_init = std::chrono::high_resolution_clock::now();
    auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_init - start_init);
    std::cout << "Grid initialization took " << init_duration.count() << " ms" << std::endl;
    
    // Print grid info
    print_grid_info();
    
    // ───────────────────────────────────────────────────────────────
    // 2) Test individual PDF evaluations
    // ───────────────────────────────────────────────────────────────
    std::cout << "\n=== Individual PDF Evaluations ===" << std::endl;
    
    double pdf_q = mass_ratio_pdf(test_inclination, q_eval);
    double pdf_vs = velocity_scale_pdf(test_inclination, vs_eval);
    double pdf_rs = radius_scale_pdf(test_inclination, rs_eval);
    double log_combined = log_mass_ratio_pdf(test_inclination, q_eval, vs_eval, rs_eval);
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "At inclination " << test_inclination << "°:" << std::endl;
    std::cout << "  PDF(q=" << q_eval << ") = " << pdf_q << std::endl;
    std::cout << "  PDF(v_s=" << vs_eval << " km/s) = " << pdf_vs << std::endl;
    std::cout << "  PDF(r_s=" << rs_eval << ") = " << pdf_rs << std::endl;
    std::cout << "  log(combined PDF) = " << log_combined << std::endl;
    std::cout << "  Expected: " << log(pdf_q) + log(pdf_vs) + log(pdf_rs) << std::endl;
    
    // ───────────────────────────────────────────────────────────────
    // 3) Performance test for combined evaluation
    // ───────────────────────────────────────────────────────────────
    std::cout << "\n=== Performance Test ===" << std::endl;
    const int n_eval = 100000;
    
    auto start_perf = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_eval; ++i) {
        double test_q = q_eval + i * 0.00001;
        double test_vs = vs_eval + i * 0.001;
        double test_rs = rs_eval + i * 0.000001;
        double log_pdf = log_mass_ratio_pdf(test_inclination, test_q, test_vs, test_rs);
    }
    auto end_perf = std::chrono::high_resolution_clock::now();
    auto perf_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_perf - start_perf);
    
    std::cout << n_eval << " combined evaluations took " << perf_duration.count() << " μs" << std::endl;
    std::cout << "Average time per evaluation: " << perf_duration.count() / (double)n_eval << " μs" << std::endl;
    
    // ───────────────────────────────────────────────────────────────
    // 4) Generate reference samples for validation
    // ───────────────────────────────────────────────────────────────
    const int nsamp = 10000;
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::normal_distribution<double> dist_m1(m1_mean, m1_err);
    std::normal_distribution<double> dist_m2(m2_mean, m2_err);
    std::normal_distribution<double> dist_K(K_mean, K_err);
    std::normal_distribution<double> dist_R(R_mean, R_err);
    std::normal_distribution<double> dist_P(P_mean, P_err);
    
    std::vector<double> ratios, v_scales, r_scales;
    ratios.reserve(nsamp);
    v_scales.reserve(nsamp);
    r_scales.reserve(nsamp);
    
    double sin_i = sin(test_inclination * M_PI / 180.0);
    
    for (int i = 0; i < nsamp; ++i) {
        double m1 = dist_m1(gen);
        double m2 = dist_m2(gen);
        double K = dist_K(gen);
        double R = dist_R(gen);
        double P = dist_P(gen);
        
        double q = mass_ratio_from_inclination(test_inclination, m1, m2);
        double v_s = (1 + 1 / q) * K / sin_i;
        double r_s = 2 * M_PI * R / (P * 86400.0 * v_s * 1/695700.0);
        
        ratios.push_back(q);
        v_scales.push_back(v_s);
        r_scales.push_back(r_s);
    }
    
    // ───────────────────────────────────────────────────────────────
    // 5) Create smooth curves with specified ranges
    // ───────────────────────────────────────────────────────────────
    const int n_curve = 300;

    // Mass ratio curve: 0 to 10
    std::vector<std::pair<double, double>> q_curve;
    for (int i = 0; i < n_curve; ++i) {
        double q = 0.0 + i * (10.0 / (n_curve - 1));
        double pdf = mass_ratio_pdf(test_inclination, q);
        q_curve.push_back({q, pdf});
    }

    // Velocity scale curve: 0 to 1000
    std::vector<std::pair<double, double>> vs_curve;
    for (int i = 0; i < n_curve; ++i) {
        double vs = 0.0 + i * (1000.0 / (n_curve - 1));
        double pdf = velocity_scale_pdf(test_inclination, vs);
        vs_curve.push_back({vs, pdf});
    }

    // Radius scale curve: 0 to 1
    std::vector<std::pair<double, double>> rs_curve;
    for (int i = 0; i < n_curve; ++i) {
        double rs = 0.0 + i * (1.0 / (n_curve - 1));
        double pdf = radius_scale_pdf(test_inclination, rs);
        rs_curve.push_back({rs, pdf});
    }
    
    // ───────────────────────────────────────────────────────────────
    // 6) Create histograms from samples
    // ───────────────────────────────────────────────────────────────
    const int nbin = 100;
    
    // Helper function to create histogram
    auto make_histogram = [nbin](const std::vector<double>& data, double min_val, double max_val) {
        std::vector<std::pair<double, double>> hist(nbin, {0.0, 0.0});
        double bin_width = (max_val - min_val) / nbin;
        
        for (int i = 0; i < nbin; ++i) {
            hist[i].first = min_val + (i + 0.5) * bin_width;
        }
        
        for (double v : data) {
            int bin = static_cast<int>((v - min_val) / bin_width);
            if (bin >= 0 && bin < nbin) {
                hist[bin].second += 1.0;
            }
        }
        
        // Normalize to density
        double n_total = data.size();
        for (int i = 0; i < nbin; ++i) {
            hist[i].second /= (n_total * bin_width);
        }
        
        return hist;
    };
    
    auto q_hist = make_histogram(ratios, 0, 10);
    auto vs_hist = make_histogram(v_scales, 0, 1000);
    auto rs_hist = make_histogram(r_scales, 0, 1);
    
    // ───────────────────────────────────────────────────────────────
    // 7) Plot all three PDFs (FIXED VERSION)
    // ───────────────────────────────────────────────────────────────
    try {
        Gnuplot gp;
        
        // Send all data as named data blocks first
        gp << "$q_hist << EOD\n";
        for (const auto& point : q_hist) {
            gp << point.first << " " << point.second << "\n";
        }
        gp << "EOD\n";
        
        gp << "$q_curve << EOD\n";
        for (const auto& point : q_curve) {
            gp << point.first << " " << point.second << "\n";
        }
        gp << "EOD\n";
        
        gp << "$vs_hist << EOD\n";
        for (const auto& point : vs_hist) {
            gp << point.first << " " << point.second << "\n";
        }
        gp << "EOD\n";
        
        gp << "$vs_curve << EOD\n";
        for (const auto& point : vs_curve) {
            gp << point.first << " " << point.second << "\n";
        }
        gp << "EOD\n";
        
        gp << "$rs_hist << EOD\n";
        for (const auto& point : rs_hist) {
            gp << point.first << " " << point.second << "\n";
        }
        gp << "EOD\n";
        
        gp << "$rs_curve << EOD\n";
        for (const auto& point : rs_curve) {
            gp << point.first << " " << point.second << "\n";
        }
        gp << "EOD\n";
        
        // Now create the multiplot
        gp << "set multiplot layout 3,1 title 'PDF Distributions at inclination=" 
           << test_inclination << "°'\n";
        
        // Mass ratio plot
        gp << "set xlabel 'Mass Ratio (q)'\n";
        gp << "set ylabel 'Density'\n";
        gp << "set grid\n";
        gp << "plot $q_hist with boxes title 'Histogram' fs solid 0.5, "
           << "$q_curve with lines lw 2 title 'Grid PDF'\n";
        
        // Velocity scale plot
        gp << "set xlabel 'Velocity Scale (km/s)'\n";
        gp << "set ylabel 'Density'\n";
        gp << "plot $vs_hist with boxes title 'Histogram' fs solid 0.5, "
           << "$vs_curve with lines lw 2 title 'Grid PDF'\n";
        
        // Radius scale plot
        gp << "set xlabel 'Radius Scale (unitless)'\n";
        gp << "set ylabel 'Density'\n";
        gp << "plot $rs_hist with boxes title 'Histogram' fs solid 0.5, "
           << "$rs_curve with lines lw 2 title 'Grid PDF'\n";
        
        gp << "unset multiplot\n";
        
        std::cout << "\nFirst plot completed successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Error with first plot: " << e.what() << std::endl;
    }
    
    // ───────────────────────────────────────────────────────────────
    // 8) Test different inclinations for all three quantities
    // ───────────────────────────────────────────────────────────────
    std::cout << "\n=== Multi-Inclination Test ===" << std::endl;
    std::vector<double> test_inclinations = {30.0, 45.0, 60.0, 75.0, 90.0};
    
    // Store curves for each inclination
    std::vector<std::vector<std::pair<double, double>>> q_incl_curves;
    std::vector<std::vector<std::pair<double, double>>> vs_incl_curves;
    std::vector<std::vector<std::pair<double, double>>> rs_incl_curves;
    
    for (double incl : test_inclinations) {
        std::cout << "\nInclination " << incl << "°:" << std::endl;
        
        // Evaluate at test points
        double pdf_q_test = mass_ratio_pdf(incl, q_eval);
        double pdf_vs_test = velocity_scale_pdf(incl, vs_eval);
        double pdf_rs_test = radius_scale_pdf(incl, rs_eval);
        
        std::cout << "  PDF(q=" << q_eval << ") = " << pdf_q_test << std::endl;
        std::cout << "  PDF(v_s=" << vs_eval << ") = " << pdf_vs_test << std::endl;
        std::cout << "  PDF(r_s=" << rs_eval << ") = " << pdf_rs_test << std::endl;
        
        // Generate curves
        std::vector<std::pair<double, double>> q_c, vs_c, rs_c;
        
        // Adjust ranges based on inclination (v_s depends strongly on sin(i))
        double sin_incl = sin(incl * M_PI / 180.0);
        double vs_center = (1+1/q_eval)*K_mean / sin_incl;
        double vs_range = (1+1/q_eval)*5 * K_err / sin_incl;
        
        // Mass ratio curves: 0 to 10
        for (int i = 0; i < n_curve; ++i) {
            double q = 0.0 + i * (10.0 / (n_curve - 1));
            double pdf = mass_ratio_pdf(incl, q);
            q_c.push_back({q, pdf});
        }
        
        // Velocity scale curves: 0 to 1000
        for (int i = 0; i < n_curve; ++i) {
            double vs = 0.0 + i * (1000.0 / (n_curve - 1));
            double pdf = velocity_scale_pdf(incl, vs);
            vs_c.push_back({vs, pdf});
        }
        
        // Radius scale curves: 0 to 1
        for (int i = 0; i < n_curve; ++i) {
            double rs = 0.0 + i * (1.0 / (n_curve - 1));
            double pdf = radius_scale_pdf(incl, rs);
            rs_c.push_back({rs, pdf});
        }
        
        q_incl_curves.push_back(q_c);
        vs_incl_curves.push_back(vs_c);
        rs_incl_curves.push_back(rs_c);
    }

    // ───────────────────────────────────────────────────────────────
    // 9) Plot inclination comparison (FIXED VERSION)
    // ───────────────────────────────────────────────────────────────
    try {
        Gnuplot gp2;
        
        // Send all inclination curves as data blocks
        for (size_t i = 0; i < test_inclinations.size(); ++i) {
            gp2 << "$q_incl" << i << " << EOD\n";
            for (const auto& point : q_incl_curves[i]) {
                gp2 << point.first << " " << point.second << "\n";
            }
            gp2 << "EOD\n";
            
            gp2 << "$vs_incl" << i << " << EOD\n";
            for (const auto& point : vs_incl_curves[i]) {
                gp2 << point.first << " " << point.second << "\n";
            }
            gp2 << "EOD\n";
            
            gp2 << "$rs_incl" << i << " << EOD\n";
            for (const auto& point : rs_incl_curves[i]) {
                gp2 << point.first << " " << point.second << "\n";
            }
            gp2 << "EOD\n";
        }
        
        gp2 << "set multiplot layout 3,1 title 'PDFs vs Inclination'\n";
        
        // Mass ratio plot
        gp2 << "set xlabel 'Mass Ratio (q)'\n";
        gp2 << "set ylabel 'Density'\n";
        gp2 << "set key top right\n";
        gp2 << "set grid\n";
        gp2 << "plot ";
        for (size_t i = 0; i < test_inclinations.size(); ++i) {
            if (i > 0) gp2 << ", ";
            gp2 << "$q_incl" << i << " with lines lw 2 title 'i=" << (int)test_inclinations[i] << "°'";
        }
        gp2 << "\n";
        
        // Velocity scale plot
        gp2 << "set xlabel 'Velocity Scale (km/s)'\n";
        gp2 << "set ylabel 'Density'\n";
        gp2 << "set logscale x\n";
        gp2 << "plot ";
        for (size_t i = 0; i < test_inclinations.size(); ++i) {
            if (i > 0) gp2 << ", ";
            gp2 << "$vs_incl" << i << " with lines lw 2 title 'i=" << (int)test_inclinations[i] << "°'";
        }
        gp2 << "\n";
        gp2 << "unset logscale\n";
        
        // Radius scale plot
        gp2 << "set xlabel 'Radius Scale (unitless)'\n";
        gp2 << "set ylabel 'Density'\n";
        gp2 << "plot ";
        for (size_t i = 0; i < test_inclinations.size(); ++i) {
            if (i > 0) gp2 << ", ";
            gp2 << "$rs_incl" << i << " with lines lw 2 title 'i=" << (int)test_inclinations[i] << "°'";
        }
        gp2 << "\n";
        
        gp2 << "unset multiplot\n";
        
        std::cout << "\nSecond plot completed successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Error with second plot: " << e.what() << std::endl;
    }
    
    // ───────────────────────────────────────────────────────────────
    // 10) Test boundary checking
    // ───────────────────────────────────────────────────────────────
    std::cout << "\n=== Boundary Test ===" << std::endl;
    
    // Test in-bounds
    bool in_bounds = check_in_bounds(test_inclination, q_eval, vs_eval, rs_eval);
    std::cout << "Test point in bounds: " << (in_bounds ? "YES" : "NO") << std::endl;
    
    // Test out-of-bounds
    bool out_bounds1 = check_in_bounds(5.0, q_eval, vs_eval, rs_eval);  // Low incl
    bool out_bounds2 = check_in_bounds(test_inclination, 10.0, vs_eval, rs_eval);  // High q
    bool out_bounds3 = check_in_bounds(test_inclination, q_eval, 1e6, rs_eval);  // High v_s
    
    std::cout << "Out of bounds tests:" << std::endl;
    std::cout << "  Low inclination (5°): " << (out_bounds1 ? "IN" : "OUT") << std::endl;
    std::cout << "  High q (10.0): " << (out_bounds2 ? "IN" : "OUT") << std::endl;
    std::cout << "  High v_s (1e6): " << (out_bounds3 ? "IN" : "OUT") << std::endl;
    

    // ───────────────────────────────────────────────────────────────
    // 12) 2-D PDF heat-maps   ( i vs q  |  i vs r_s )
    // ───────────────────────────────────────────────────────────────
    try
    {
        Gnuplot gp3;

        /* ----------------------------------------------------------------
           Build a reasonably fine grid in (inclination,q) and (inclination,r_s)
           that covers the *full* range that is actually tabulated in the
           MassRatioPDFGrid.  We do not have direct access to the internal
           min / max – but they are printed by print_grid_info(), so just
           copy them here (or change them by hand if you adjust the grid).
           ---------------------------------------------------------------- */
        const double I_MIN = 15.0;   // deg   (must match initialise-call!)
        const double I_MAX = 90.0;   // deg
        const double Q_MIN = 0.7;    // good generic range; outside grid gives 1e-12
        const double Q_MAX = 3.5;
        const double RS_MIN = 0.07;
        const double RS_MAX = 0.15; // 0.01 ≃ 1 % of a solar radius / a~few days

        const int NI  = 500;         // same density that the grid itself uses
        const int NQ  = 500;         // enough to make the map smooth
        const int NRS = 500;

        /* -------------- build inclination vector (inclusive end points) */
        std::vector<double> I_vals(NI);
        for (int k = 0; k < NI; ++k)
            I_vals[k] = I_MIN + k * (I_MAX - I_MIN) / (NI - 1);

        std::vector<double> Q_vals(NQ);
        for (int k = 0; k < NQ; ++k)
            Q_vals[k] = Q_MIN + k * (Q_MAX - Q_MIN) / (NQ - 1);

        std::vector<double> RS_vals(NRS);
        for (int k = 0; k < NRS; ++k)
            RS_vals[k] = RS_MIN + k * (RS_MAX - RS_MIN) / (NRS - 1);

        /* ------------------  data block :  i  vs  q  ------------------- */
        gp3 << "$IQ_PDF << EOD\n";
        for (double inc : I_vals)
        {
            for (double q : Q_vals)
            {
                double pdf = mass_ratio_pdf(inc, q);
                gp3 << inc << " " << q << " " << pdf << "\n";
            }
            gp3 << "\n";                        // blank line → new scanline
        }
        gp3 << "EOD\n";

        /* ------------------  data block :  i  vs  r_s  ----------------- */
        gp3 << "$IRS_PDF << EOD\n";
        for (double inc : I_vals)
        {
            for (double rs : RS_vals)
            {
                double pdf = radius_scale_pdf(inc, rs);
                gp3 << inc << " " << rs << " " << pdf << "\n";
            }
            gp3 << "\n";
        }
        gp3 << "EOD\n";

        /* ------------------------- plotting ---------------------------- */
        gp3 << "set term qt size 1000,450\n";
        gp3 << "set multiplot layout 1,2 title '2-D PDF maps (log colour-scale)'\n";

        // ----- panel 1 :  i  vs q
        gp3 << "set xlabel 'Inclination  i  [deg]'\n";
        gp3 << "set ylabel 'Mass ratio  q'\n";
        gp3 << "set xrange [" << I_MIN << ":" << I_MAX << "]\n";
        gp3 << "set yrange [" << Q_MIN << ":" << Q_MAX << "]\n";
        //gp3 << "set logscale cb\n";
        gp3 << "set cbrange [1e-2:0.7]\n";        // <-- colour scale only 10⁻4 … 2
        gp3 << "set cblabel 'PDF(i,q)'\n";
        gp3 << "unset key\n";
        gp3 << "set pm3d map\n";
        gp3 << "splot $IQ_PDF using 1:2:3 notitle\n";

        // ----- panel 2 :  i  vs r_s
        gp3 << "set xlabel 'Inclination  i  [deg]'\n";
        gp3 << "set ylabel 'Radius-scale  r_s'\n";
        gp3 << "set xrange [" << I_MIN << ":" << I_MAX << "]\n";
        gp3 << "set yrange [" << RS_MIN << ":" << RS_MAX << "]\n";
        gp3 << "set cbrange [0.1:100.0]\n";        // INDEPENDENT colorbar range for PDF(i,r_s)
        gp3 << "set cblabel 'PDF(i,r_s)'\n";
        gp3 << "splot $IRS_PDF using 1:2:3 notitle\n";

        gp3 << "unset multiplot\n";

        std::cout << "\n2-D heat-maps shown – look for horizontal banding along i.\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "[2-D plot ERROR] " << e.what() << std::endl;
    }


    // ───────────────────────────────────────────────────────────────
    // 11) Statistical summary
    // ───────────────────────────────────────────────────────────────
    std::cout << "\n=== Statistical Summary ===" << std::endl;
    
    // Compute statistics for samples
    auto compute_stats = [](const std::vector<double>& data) {
        double sum = std::accumulate(data.begin(), data.end(), 0.0);
        double mean = sum / data.size();
        double sq_sum = 0.0;
        for (double v : data) {
            sq_sum += (v - mean) * (v - mean);
        }
        double stdev = std::sqrt(sq_sum / (data.size() - 1));
        
        auto minmax = std::minmax_element(data.begin(), data.end());
        
        return std::make_tuple(mean, stdev, *minmax.first, *minmax.second);
    };
    
    auto [q_mean, q_std, q_min_s, q_max_s] = compute_stats(ratios);
    auto [vs_mean, vs_std, vs_min_s, vs_max_s] = compute_stats(v_scales);
    auto [rs_mean, rs_std, rs_min_s, rs_max_s] = compute_stats(r_scales);
    
    std::cout << "Mass ratio statistics:" << std::endl;
    std::cout << "  Mean: " << q_mean << ", Std: " << q_std << std::endl;
    std::cout << "  Range: [" << q_min_s << ", " << q_max_s << "]" << std::endl;
    
    std::cout << "Velocity scale statistics:" << std::endl;
    std::cout << "  Mean: " << vs_mean << " km/s, Std: " << vs_std << " km/s" << std::endl;
    std::cout << "  Range: [" << vs_min_s << ", " << vs_max_s << "] km/s" << std::endl;
    
    std::cout << "Radius scale statistics:" << std::endl;
    std::cout << "  Mean: " << rs_mean << ", Std: " << rs_std << std::endl;
    std::cout << "  Range: [" << rs_min_s << ", " << rs_max_s << "]" << std::endl;
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    std::cout << "Press Enter to exit..." << std::endl;
    std::cin.get();
    
    // Clean up
    cleanup_mass_ratio_pdf_grid();
    
    return 0;
}