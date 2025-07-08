#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <iostream>
#include "mass_ratio_pdf.h"
#include "lcurve_base/lcurve.h"

double day_to_sec = 86400.0;
double km_to_solrad = 1/695700.0;

MassRatioPDFGrid* g_mass_ratio_grid = nullptr;
double bisect(
    const function<double(double)> &f,
    double c_lo,
    double c_hi,
    double tol = 1e-9,
    int max_iter = 100
) {
    double f_lo = f(c_lo), f_hi = f(c_hi);
    if (f_lo * f_hi > 0)
        throw runtime_error("Bisection requires f(c_lo) and f(c_hi) of opposite signs.");

    double c_mid = 0, f_mid;
    for (int i = 0; i < max_iter; ++i) {
        c_mid = 0.5 * (c_lo + c_hi);
        f_mid = f(c_mid);
        if (abs(f_mid) < tol || 0.5 * (c_hi - c_lo) < tol)
            return c_mid;
        if (f_lo * f_mid <= 0) {
            c_hi = c_mid;
            f_hi = f_mid;
        } else {
            c_lo = c_mid;
            f_lo = f_mid;
        }
    }
    return c_mid; // may not converge fully
}

// Numerically computed mass ratio (Mass ratio errors are way too low to be relevant)
double mass_ratio_from_inclination(double inclination, double mass1, double min_mass2) {
    // Precompute constant factor A = a^3/(a+b)^2
    double A = pow(min_mass2, 3) / pow(min_mass2 + mass1, 2);

    // Compute target = sin^3(k)
    double inclination_rad = inclination * M_PI / 180.0;
    double target = pow(sin(inclination_rad), 3);

    // Define f(c) = A * (c+b)^2 / c^3 - target
    auto f = [&](double c) {
        return A * pow(c + mass1, 2) / pow(c, 3) - target;
    };

    // Pick a bracket [c_lo, c_hi] such that f(c_lo)*f(c_hi) < 0.
    double c_lo = 1e-6;
    double c_hi = 1e5;
    double c_sol = 0.0;
    try {
        c_sol = bisect(f, c_lo, c_hi);
    } catch (const exception &e) {
        cerr << "Error: " << e.what() << "\n";
        return 1e-12;
    }

    return c_sol / mass1;
}

class MassRatioPDFGrid {
private:
    // Grid parameters
    std::vector<double> inclination_grid;
    std::vector<double> q_grid;
    std::vector<double> vs_grid;
    std::vector<double> rs_grid;
    
    std::vector<std::vector<double>> pdf_grid_q;   // pdf_grid_q[i_incl][i_q]
    std::vector<std::vector<double>> pdf_grid_vs;  // pdf_grid_vs[i_incl][i_vs]
    std::vector<std::vector<double>> pdf_grid_rs;  // pdf_grid_rs[i_incl][i_rs]
    
    double incl_min, incl_max, incl_dx;
    double q_min, q_max, q_dx;
    double vs_min, vs_max, vs_dx;
    double rs_min, rs_max, rs_dx;
    int n_incl, n_q, n_vs, n_rs;
    
    // Mass distribution parameters (fixed for this grid)
    double m1_mean, m1_err, m2_mean, m2_err;
    
    // Velocity scale parameters
    double K_mean, K_err;
    
    // Radius scale parameters
    double R_mean, R_err;
    double P_mean, P_err;
    
public:
    MassRatioPDFGrid(
        double m1_mean_, double m1_err_,
        double m2_mean_, double m2_err_,
        double K_mean_, double K_err_,
        double R_mean_, double R_err_,
        double P_mean_, double P_err_,
        double incl_min_, double incl_max_, int n_incl_,
        int n_q_, int n_vs_, int n_rs_,
        int nsamp 
    ) : m1_mean(m1_mean_), m1_err(m1_err_), m2_mean(m2_mean_), m2_err(m2_err_),
        K_mean(K_mean_), K_err(K_err_), R_mean(R_mean_), R_err(R_err_),
        P_mean(P_mean_), P_err(P_err_),
        incl_min(incl_min_), incl_max(incl_max_), n_incl(n_incl_), 
        n_q(n_q_), n_vs(n_vs_), n_rs(n_rs_) {
        
        // ANSI color codes
        const string RESET = "\033[0m";
        const string BRIGHT_BLUE = "\033[94m";
        const string BRIGHT_GREEN = "\033[92m";
        const string BRIGHT_CYAN = "\033[96m";
        const string DIM = "\033[2m";
        
        cout << BRIGHT_BLUE << "   Pre-computing mass ratio, velocity scale, and radius scale PDF grids..." << RESET << endl;
        cout << DIM << "   Inclination: [" << incl_min << "°, " << incl_max << "°] × " << n_incl << " points" << RESET << endl;
        
        // Set up inclination grid
        incl_dx = (incl_max - incl_min) / (n_incl - 1);
        inclination_grid.resize(n_incl);
        for (int i = 0; i < n_incl; ++i) {
            inclination_grid[i] = incl_min + i * incl_dx;
        }
        
        // Generate samples for all inclinations to determine global ranges
        random_device rd;
        default_random_engine gen(rd());
        normal_distribution<double> dist_m1(m1_mean, m1_err);
        normal_distribution<double> dist_m2(m2_mean, m2_err);
        normal_distribution<double> dist_K(K_mean, K_err);
        normal_distribution<double> dist_R(R_mean, R_err);
        normal_distribution<double> dist_P(P_mean, P_err);
        
        vector<double> all_ratios;
        vector<double> all_vs;
        vector<double> all_rs;
        all_ratios.reserve(nsamp * n_incl);
        all_vs.reserve(nsamp * n_incl);
        all_rs.reserve(nsamp * n_incl);
        
        // Sample across all inclinations to get global bounds
        for (int i_incl = 0; i_incl < n_incl; ++i_incl) {
            double incl = inclination_grid[i_incl];
            double sin_i = sin(incl * M_PI / 180.0);
            
            // Skip very small inclinations to avoid division by zero
            if (sin_i < 1e-6) continue;
            
            for (int i = 0; i < nsamp; ++i) {
                double m1 = dist_m1(gen);
                double m2 = dist_m2(gen);
                double K = dist_K(gen);
                double R = dist_R(gen);
                double P = dist_P(gen);
                
                double q = mass_ratio_from_inclination(incl, m1, m2);
                double v_s = (1+1/q) * K / sin_i;
                double r_s = 2*M_PI*R / (P*day_to_sec * v_s* km_to_solrad);
                
                all_ratios.push_back(q);
                all_vs.push_back(v_s);
                all_rs.push_back(r_s);
            }
        }
        
        // Determine global ranges (99.5% coverage)
        sort(all_ratios.begin(), all_ratios.end());
        sort(all_vs.begin(), all_vs.end());
        sort(all_rs.begin(), all_rs.end());
        
        int q_idx_low = static_cast<int>(0.0025 * all_ratios.size());
        int q_idx_high = static_cast<int>(0.9925 * all_ratios.size());
        
        int vs_idx_low = static_cast<int>(0.0025 * all_vs.size());
        int vs_idx_high = static_cast<int>(0.9925 * all_vs.size());

        int rs_idx_low = static_cast<int>(0.0025 * all_rs.size());
        int rs_idx_high = static_cast<int>(0.9925 * all_rs.size());

        q_min = all_ratios[q_idx_low];
        q_max = all_ratios[q_idx_high];
        q_dx = (q_max - q_min) / (n_q - 1);
        
        vs_min = all_vs[vs_idx_low];
        vs_max = all_vs[vs_idx_high];
        vs_dx = (vs_max - vs_min) / (n_vs - 1);
        
        rs_min = all_rs[rs_idx_low];
        rs_max = all_rs[rs_idx_high];
        rs_dx = (rs_max - rs_min) / (n_rs - 1);
        
        cout << DIM << "   Mass ratio: [" << fixed << setprecision(3) << q_min << ", " << q_max << "] × " << n_q << " points" << RESET << endl;
        cout << DIM << "   Velocity scale: [" << fixed << setprecision(1) << vs_min << ", " << vs_max << "] km/s × " << n_vs << " points" << RESET << endl;
        cout << DIM << "   Radius scale: [" << fixed << setprecision(4) << rs_min << ", " << rs_max << "] × " << n_rs << " points" << RESET << endl;
        
        // Set up grids
        q_grid.resize(n_q);
        vs_grid.resize(n_vs);
        rs_grid.resize(n_rs);
        
        for (int i = 0; i < n_q; ++i) {
            q_grid[i] = q_min + i * q_dx;
        }
        for (int i = 0; i < n_vs; ++i) {
            vs_grid[i] = vs_min + i * vs_dx;
        }
        for (int i = 0; i < n_rs; ++i) {
            rs_grid[i] = rs_min + i * rs_dx;
        }
        
        // Initialize PDF grids
        pdf_grid_q.resize(n_incl);
        pdf_grid_vs.resize(n_incl);
        pdf_grid_rs.resize(n_incl);
        for (int i = 0; i < n_incl; ++i) {
            pdf_grid_q[i].resize(n_q, 0.0);
            pdf_grid_vs[i].resize(n_vs, 0.0);
            pdf_grid_rs[i].resize(n_rs, 0.0);
        }
        
        // Compute PDFs for each inclination
        for (int i_incl = 0; i_incl < n_incl; ++i_incl) {
            double incl = inclination_grid[i_incl];
            double sin_i = sin(incl * M_PI / 180.0);
            
            // Simple progress indicator every 20%
            if (i_incl % (n_incl / 5) == 0 && i_incl > 0) {
                int percent = (i_incl * 100) / n_incl;
                cout << "\r" << BRIGHT_CYAN << "   Progress: " << percent << "% (" << i_incl << "/" << n_incl << ")" << RESET;
                cout.flush();
            }
            
            // Skip very small inclinations
            if (sin_i < 1e-6) continue;
            
            // Generate samples for this inclination
            vector<double> ratios, v_scales, r_scales;
            ratios.reserve(nsamp);
            v_scales.reserve(nsamp);
            r_scales.reserve(nsamp);
            
            for (int i = 0; i < nsamp; ++i) {
                double m1 = dist_m1(gen);
                double m2 = dist_m2(gen);
                double K = dist_K(gen);
                double R = dist_R(gen);
                double P = dist_P(gen);
                
                double q = mass_ratio_from_inclination(incl, m1, m2);
                double v_s = (1 + 1 / q) * K / sin_i;
                double r_s = 2 * M_PI * R / (P * day_to_sec * v_s* km_to_solrad);
                
                ratios.push_back(q);
                v_scales.push_back(v_s);
                r_scales.push_back(r_s);
            }
            
            // Compute KDE bandwidths
            auto compute_bandwidth = [](const vector<double>& data, int n) {
                double sum = accumulate(data.begin(), data.end(), 0.0);
                double mean = sum / n;
                double sq_sum = 0.0;
                for (double v : data) {
                    sq_sum += (v - mean) * (v - mean);
                }
                double sigma = sqrt(sq_sum / (n - 1));
                return 1.06 * sigma * pow(static_cast<double>(n), -0.2);
            };
            
            double h_q = compute_bandwidth(ratios, nsamp);
            double h_vs = compute_bandwidth(v_scales, nsamp);
            double h_rs = compute_bandwidth(r_scales, nsamp);
            
            // Compute PDFs using KDE
            const double inv_sqrt_2pi = 1.0 / sqrt(2.0 * M_PI);
            
            // Mass ratio PDF
            for (int i_q = 0; i_q < n_q; ++i_q) {
                double q = q_grid[i_q];
                double pdf_val = 0.0;
                
                for (double xi : ratios) {
                    double u = (q - xi) / h_q;
                    pdf_val += exp(-0.5 * u * u) * inv_sqrt_2pi;
                }
                pdf_grid_q[i_incl][i_q] = pdf_val / (nsamp * h_q);
            }
            
            // Velocity scale PDF
            for (int i_vs = 0; i_vs < n_vs; ++i_vs) {
                double vs = vs_grid[i_vs];
                double pdf_val = 0.0;
                
                for (double xi : v_scales) {
                    double u = (vs - xi) / h_vs;
                    pdf_val += exp(-0.5 * u * u) * inv_sqrt_2pi;
                }
                pdf_grid_vs[i_incl][i_vs] = pdf_val / (nsamp * h_vs);
            }
            
            // Radius scale PDF
            for (int i_rs = 0; i_rs < n_rs; ++i_rs) {
                double rs = rs_grid[i_rs];
                double pdf_val = 0.0;
                
                for (double xi : r_scales) {
                    double u = (rs - xi) / h_rs;
                    pdf_val += exp(-0.5 * u * u) * inv_sqrt_2pi;
                }
                pdf_grid_rs[i_incl][i_rs] = pdf_val / (nsamp * h_rs);
            }
        }
        
        cout << "\r" << BRIGHT_GREEN << "✓ Grid pre-computation complete!          " << RESET << endl;
    }
    
    // Fast PDF evaluation using bilinear interpolation
    double evaluate_q(double inclination, double q) const {
        // Handle out-of-bounds cases
        if (inclination < incl_min || inclination > incl_max || 
            q < q_min || q > q_max) {
            return 1e-12;
        }
        
        // Find grid positions
        double incl_pos = (inclination - incl_min) / incl_dx;
        double q_pos = (q - q_min) / q_dx;
        
        int i_incl = static_cast<int>(incl_pos);
        int i_q = static_cast<int>(q_pos);
        
        // Boundary checks
        if (i_incl >= n_incl - 1) i_incl = n_incl - 2;
        if (i_q >= n_q - 1) i_q = n_q - 2;
        if (i_incl < 0) i_incl = 0;
        if (i_q < 0) i_q = 0;
        
        // Bilinear interpolation
        double frac_incl = incl_pos - i_incl;
        double frac_q = q_pos - i_q;
        
        double pdf_00 = pdf_grid_q[i_incl][i_q];
        double pdf_10 = pdf_grid_q[i_incl + 1][i_q];
        double pdf_01 = pdf_grid_q[i_incl][i_q + 1];
        double pdf_11 = pdf_grid_q[i_incl + 1][i_q + 1];
        
        double pdf_0 = pdf_00 * (1.0 - frac_incl) + pdf_10 * frac_incl;
        double pdf_1 = pdf_01 * (1.0 - frac_incl) + pdf_11 * frac_incl;
        
        return pdf_0 * (1.0 - frac_q) + pdf_1 * frac_q;
    }
    
    double evaluate_vs(double inclination, double vs) const {
        if (inclination < incl_min || inclination > incl_max || 
            vs < vs_min || vs > vs_max) {
            return 1e-12;
        }
        
        double incl_pos = (inclination - incl_min) / incl_dx;
        double vs_pos = (vs - vs_min) / vs_dx;
        
        int i_incl = static_cast<int>(incl_pos);
        int i_vs = static_cast<int>(vs_pos);
        
        if (i_incl >= n_incl - 1) i_incl = n_incl - 2;
        if (i_vs >= n_vs - 1) i_vs = n_vs - 2;
        if (i_incl < 0) i_incl = 0;
        if (i_vs < 0) i_vs = 0;
        
        double frac_incl = incl_pos - i_incl;
        double frac_vs = vs_pos - i_vs;
        
        double pdf_00 = pdf_grid_vs[i_incl][i_vs];
        double pdf_10 = pdf_grid_vs[i_incl + 1][i_vs];
        double pdf_01 = pdf_grid_vs[i_incl][i_vs + 1];
        double pdf_11 = pdf_grid_vs[i_incl + 1][i_vs + 1];
        
        double pdf_0 = pdf_00 * (1.0 - frac_incl) + pdf_10 * frac_incl;
        double pdf_1 = pdf_01 * (1.0 - frac_incl) + pdf_11 * frac_incl;
        
        return pdf_0 * (1.0 - frac_vs) + pdf_1 * frac_vs;
    }
    
    double evaluate_rs(double inclination, double rs) const {
        if (inclination < incl_min || inclination > incl_max || 
            rs < rs_min || rs > rs_max) {
            return 1e-12;
        }
        
        double incl_pos = (inclination - incl_min) / incl_dx;
        double rs_pos = (rs - rs_min) / rs_dx;
        
        int i_incl = static_cast<int>(incl_pos);
        int i_rs = static_cast<int>(rs_pos);
        
        if (i_incl >= n_incl - 1) i_incl = n_incl - 2;
        if (i_rs >= n_rs - 1) i_rs = n_rs - 2;
        if (i_incl < 0) i_incl = 0;
        if (i_rs < 0) i_rs = 0;
        
        double frac_incl = incl_pos - i_incl;
        double frac_rs = rs_pos - i_rs;
        
        double pdf_00 = pdf_grid_rs[i_incl][i_rs];
        double pdf_10 = pdf_grid_rs[i_incl + 1][i_rs];
        double pdf_01 = pdf_grid_rs[i_incl][i_rs + 1];
        double pdf_11 = pdf_grid_rs[i_incl + 1][i_rs + 1];
        
        double pdf_0 = pdf_00 * (1.0 - frac_incl) + pdf_10 * frac_incl;
        double pdf_1 = pdf_01 * (1.0 - frac_incl) + pdf_11 * frac_incl;
        
        return pdf_0 * (1.0 - frac_rs) + pdf_1 * frac_rs;
    }
    
    // Combined log PDF evaluation for MCMC
    double log_combined_pdf(double inclination, double q, double v_s, double r_s) const {
        double pdf_q = evaluate_q(inclination, q);
        double pdf_vs = evaluate_vs(inclination, v_s);
        double pdf_rs = evaluate_rs(inclination, r_s);
        
        // Return sum of log PDFs
        return log(pdf_q) + log(pdf_vs) + log(pdf_rs);
    }
    
    // For debugging
    bool in_bounds(double inclination, double q, double v_s, double r_s) const {
        return (inclination >= incl_min && inclination <= incl_max &&
                q >= q_min && q <= q_max &&
                v_s >= vs_min && v_s <= vs_max &&
                r_s >= rs_min && r_s <= rs_max);
    }
    
    // Get grid info
    void print_info() const {
        std::cout << "Grid info:" << std::endl;
        std::cout << "  Inclination: [" << incl_min << ", " << incl_max 
                  << "] deg, " << n_incl << " points" << std::endl;
        std::cout << "  Mass ratio: [" << q_min << ", " << q_max 
                  << "], " << n_q << " points" << std::endl;
        std::cout << "  Velocity scale: [" << vs_min << ", " << vs_max 
                  << "] km/s, " << n_vs << " points" << std::endl;
        std::cout << "  Radius scale: [" << rs_min << ", " << rs_max 
                  << "], " << n_rs << " points" << std::endl;
        std::cout << "  Mass distributions: m1~N(" << m1_mean << ", " << m1_err 
                  << "), m2~N(" << m2_mean << ", " << m2_err << ")" << std::endl;
        std::cout << "  K distribution: K~N(" << K_mean << ", " << K_err << ") km/s" << std::endl;
        std::cout << "  R distribution: R~N(" << R_mean << ", " << R_err << ") R_sun" << std::endl;
        std::cout << "  P distribution: P~N(" << P_mean << ", " << P_err << ") days" << std::endl;
    }
};

// Global grid instance
extern MassRatioPDFGrid* g_mass_ratio_grid;

// Function to initialize the grid (call once at start of program)
void initialize_mass_ratio_pdf_grid(
    double m1_mean, double m1_err,
    double m2_mean, double m2_err,
    double K_mean, double K_err,
    double R_mean, double R_err,
    double P_mean, double P_err,
    double incl_min, double incl_max, int n_incl,
    int n_q, int n_vs, int n_rs, int nsamp
) {
    if (g_mass_ratio_grid != nullptr) {
        delete g_mass_ratio_grid;
    }
    
    g_mass_ratio_grid = new MassRatioPDFGrid(
        m1_mean, m1_err, m2_mean, m2_err,
        K_mean, K_err, R_mean, R_err, P_mean, P_err,
        incl_min, incl_max, n_incl, n_q, n_vs, n_rs, nsamp
    );
}

// Main function for MCMC: returns sum of log PDFs
double log_mass_ratio_pdf(double inclination, double q, double v_s, double r_s) {
    if (g_mass_ratio_grid == nullptr) {
        throw std::runtime_error("Mass ratio PDF grid not initialized! Call initialize_mass_ratio_pdf_grid() first.");
    }
    
    return g_mass_ratio_grid->log_combined_pdf(inclination, q, v_s, r_s);
}

// Individual PDF access functions (if needed)
double mass_ratio_pdf(double inclination, double q) {
    if (g_mass_ratio_grid == nullptr) {
        throw std::runtime_error("Mass ratio PDF grid not initialized!");
    }
    return g_mass_ratio_grid->evaluate_q(inclination, q);
}

double velocity_scale_pdf(double inclination, double v_s) {
    if (g_mass_ratio_grid == nullptr) {
        throw std::runtime_error("Mass ratio PDF grid not initialized!");
    }
    return g_mass_ratio_grid->evaluate_vs(inclination, v_s);
}

double radius_scale_pdf(double inclination, double r_s) {
    if (g_mass_ratio_grid == nullptr) {
        throw std::runtime_error("Mass ratio PDF grid not initialized!");
    }
    return g_mass_ratio_grid->evaluate_rs(inclination, r_s);
}

// Cleanup function (call at end of program)
void cleanup_mass_ratio_pdf_grid() {
    if (g_mass_ratio_grid != nullptr) {
        delete g_mass_ratio_grid;
        g_mass_ratio_grid = nullptr;
    }
}

// Additional utility functions for testing
void print_grid_info() {
    if (g_mass_ratio_grid != nullptr) {
        g_mass_ratio_grid->print_info();
    }
}

bool check_in_bounds(double inclination, double q, double v_s, double r_s) {
    if (g_mass_ratio_grid == nullptr) {
        return false;
    }
    return g_mass_ratio_grid->in_bounds(inclination, q, v_s, r_s);
}