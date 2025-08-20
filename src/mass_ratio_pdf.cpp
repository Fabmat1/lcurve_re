/*****************************************************************************************
 Mass–ratio / velocity–scale / radius–scale PDF grid
 ─────────────────────────────────────────────────────────────────────────────────────────
 Now with an “outlier-strip” clean-up:     (fixes ‘banding’ arising from bad KDEs)

 1.  After the first pass over all inclination strips we measure, for every strip i,
     the mean absolute difference to its two neighbours   ⟨|pdf(i)-pdf(i±1)|⟩.
 2.  From the resulting distribution we compute  μ  and  σ  and flag every strip with
     a divergence > μ+3σ in *any* of the three grids ( q / v_s / r_s ).
 3.  Flagged strips are regenerated (fresh Monte-Carlo sampling + KDE).
 4.  We recompute the divergences and print a short report.

 If a pathological strip stubbornly stays above 3σ it is kept (with a warning),
 but in practise a single regeneration is sufficient.

 The public interface is untouched: simply re-compile and link this file instead of
 the previous version and call  initialize_mass_ratio_pdf_grid(…)  as before.
*****************************************************************************************/

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <functional>
#include <limits>
#include <sstream>

#include "mass_ratio_pdf.h"
#include "lcurve_base/lcurve.h"

/* ─────────────────────────── constants ──────────────────────────── */
static constexpr double day_to_sec   = 86400.0;
static constexpr double km_to_solrad = 1.0 / 695700.0;

/* ─────────────────────────── globals  ───────────────────────────── */
MassRatioPDFGrid* g_mass_ratio_grid = nullptr;

/* ─────────────────────────  helpers  ────────────────────────────── */
double bisect(const std::function<double(double)>& f,
              double c_lo, double c_hi,
              double tol = 1e-9, int max_iter = 100)
{
    double f_lo = f(c_lo), f_hi = f(c_hi);
    if (f_lo * f_hi > 0.0)
        throw std::runtime_error("Bisection requires f(c_lo) and f(c_hi) of opposite signs.");

    double c_mid = 0.0, f_mid;
    for (int i = 0; i < max_iter; ++i) {
        c_mid = 0.5 * (c_lo + c_hi);
        f_mid = f(c_mid);
        if (std::fabs(f_mid) < tol || 0.5 * (c_hi - c_lo) < tol) return c_mid;
        if (f_lo * f_mid <= 0.0) { c_hi = c_mid; f_hi = f_mid; }
        else                     { c_lo = c_mid; f_lo = f_mid; }
    }
    return c_mid;        // may not be fully converged – caller beware
}

/* invert m sin³i  →  mass ratio -------------------------------------------------------- */
double mass_ratio_from_inclination(double inclination_deg,
                                   double mass1, double min_mass2)
{
    const double A = std::pow(min_mass2, 3) / std::pow(min_mass2 + mass1, 2);
    const double target = std::pow(std::sin(inclination_deg * M_PI / 180.0), 3);

    auto f = [&](double c) { return A * std::pow(c + mass1, 2) / std::pow(c, 3) - target; };

    try { return bisect(f, 1e-6, 1e5) / mass1; }
    catch (const std::exception& e) {
        std::cerr << "\n[Mass-ratio inversion ERROR]\n"
                  << "  inclination  : " << inclination_deg << " deg\n"
                  << "  m1 (primary) : " << mass1          << " M☉\n"
                  << "  m2,min       : " << min_mass2       << " M☉\n"
                  << "  message      : " << e.what()        << '\n';
        return -1.0;
    }
}

/* ───────────────── Scott’s bandwidth helper ────────────────────── */
inline double scott_bandwidth(const std::vector<double>& data)
{
    const std::size_t n = data.size();
    if (n < 2) return 0.0;
    const double mean = std::accumulate(data.begin(), data.end(), 0.0) / n;
    double var = 0.0;
    for (double v : data) var += (v - mean) * (v - mean);
    var /= static_cast<double>(n - 1);
    const double sigma = std::sqrt(var);
    return 1.06 * sigma * std::pow(static_cast<double>(n), -0.2);
}

/* ────────────────────────  Class definition  ───────────────────── */
class MassRatioPDFGrid {
private:
    /* ───── grid axes ───── */
    std::vector<double> inclination_grid;             // n_incl
    std::vector<double> q_grid;                       // n_q
    std::vector<double> vs_grid;                      // n_vs
    std::vector<double> rs_grid;                      // n_rs

    /* ───── PDF values ──── pdf_grid_x[ i_incl ][ i_axis ] */
    std::vector<std::vector<double>> pdf_grid_q;
    std::vector<std::vector<double>> pdf_grid_vs;
    std::vector<std::vector<double>> pdf_grid_rs;

    /* ───── axis ranges & steps ───── */
    double incl_min, incl_max, incl_dx;
    double q_min,    q_max,    q_dx;
    double vs_min,   vs_max,   vs_dx;
    double rs_min,   rs_max,   rs_dx;
    int    n_incl, n_q, n_vs, n_rs;

    /* ───── population hyper-parameters (kept for info) ───── */
    double m1_mean, m1_err, m2_mean, m2_err;
    double K_mean,  K_err;
    double R_mean,  R_err;
    double P_mean,  P_err;

    /* ──────────────────────────────────────────────────────── */
    /* helper: compute strip divergence for one grid           */
    static void compute_divergence(const std::vector<std::vector<double>>& grid,
                                   std::vector<double>& out)
    {
        const int n_incl = static_cast<int>(grid.size());
        const int n_axis = static_cast<int>(grid.front().size());
        out.assign(n_incl, 0.0);

        for (int i = 1; i < n_incl - 1; ++i) {
            double diff_sum = 0.0;
            for (int j = 0; j < n_axis; ++j) {
                diff_sum += 0.5 * (std::fabs(grid[i][j] - grid[i - 1][j]) +
                                   std::fabs(grid[i][j] - grid[i + 1][j]));
            }
            out[i] = diff_sum / n_axis;
        }
    }

public:
    MassRatioPDFGrid(
        double m1_mean_, double m1_err_,
        double m2_mean_, double m2_err_,
        double K_mean_,  double K_err_,
        double R_mean_,  double R_err_,
        double P_mean_,  double P_err_,
        double incl_min_, double incl_max_, int n_incl_,
        int n_q_, int n_vs_, int n_rs_,
        int nsamp)
    : m1_mean(m1_mean_), m1_err(m1_err_), m2_mean(m2_mean_), m2_err(m2_err_),
      K_mean(K_mean_),   K_err(K_err_),   R_mean(R_mean_),   R_err(R_err_),
      P_mean(P_mean_),   P_err(P_err_),
      incl_min(incl_min_), incl_max(incl_max_), n_incl(n_incl_),
      n_q(n_q_), n_vs(n_vs_), n_rs(n_rs_)
    {
        /* Some ANSI colours for nice console output ---------------------------------- */
        const std::string RESET        = "\033[0m";
        const std::string BRIGHT_BLUE  = "\033[94m";
        const std::string BRIGHT_GREEN = "\033[92m";
        const std::string BRIGHT_CYAN  = "\033[96m";
        const std::string DIM          = "\033[2m";

        std::cout << BRIGHT_BLUE << "   Pre-computing PDF grid ..." << RESET << '\n';
        std::cout << DIM
                  << "   Inclination: [" << incl_min << "°, " << incl_max << "°] × "
                  << n_incl << " points" << RESET << '\n';

        /* ─── set-up inclination axis ─────────────────────────────────────────────── */
        incl_dx = (incl_max - incl_min) / double(n_incl - 1);
        inclination_grid.resize(n_incl);
        for (int i = 0; i < n_incl; ++i) inclination_grid[i] = incl_min + i * incl_dx;

        /* ─── random generators ──────────────────────────────────────────────────── */
        std::random_device rd;
        std::default_random_engine gen(rd());
        std::normal_distribution<double> dist_m1(m1_mean, m1_err);
        std::normal_distribution<double> dist_m2(m2_mean, m2_err);
        std::normal_distribution<double> dist_K (K_mean,  K_err);
        std::normal_distribution<double> dist_R (R_mean,  R_err);
        std::normal_distribution<double> dist_P (P_mean,  P_err);

        /* helper: draw strictly positive numbers ----------------------------------- */
        auto sample_pos = [&](auto& dist, int max_iter = 10000) -> double {
            for (int i = 0; i < max_iter; ++i) {
                double v = dist(gen);
                if (v > 0.0 && std::isfinite(v)) return v;
            }
            std::ostringstream oss;
            oss << "[sample_pos ERROR] Could not draw positive value after "
                << max_iter << " tries.  μ=" << dist.mean()
                << " σ=" << dist.stddev();
            throw std::runtime_error(oss.str());
        };

        /* ───────────────────── global ranges (use a first MC sweep) ─────────────── */
        std::vector<double> all_q, all_vs, all_rs;
        all_q.reserve (nsamp * n_incl);
        all_vs.reserve(nsamp * n_incl);
        all_rs.reserve(nsamp * n_incl);

        for (double incl : inclination_grid) {
            const double sin_i = std::sin(incl * M_PI / 180.0);
            if (sin_i < 1e-6) continue;

            for (int i = 0; i < nsamp; ++i) {
                const double m1 = sample_pos(dist_m1);
                const double m2 = sample_pos(dist_m2);
                const double K  = sample_pos(dist_K );
                const double R  = sample_pos(dist_R );
                const double P  = sample_pos(dist_P );

                const double q  = mass_ratio_from_inclination(incl, m1, m2);
                const double v_s = (1.0 + 1.0 / q) * K / sin_i;
                const double r_s = 2.0 * M_PI * R /
                                   (P * day_to_sec * v_s * km_to_solrad);

                if (q  <= 0.0 || v_s <= 0.0 || r_s <= 0.0 ||
                    !std::isfinite(q) || !std::isfinite(v_s) || !std::isfinite(r_s))
                    continue;

                all_q .push_back(q );
                all_vs.push_back(v_s);
                all_rs.push_back(r_s);
            }
        }

        auto ranged = [](std::vector<double>& v, double low, double high) {
            std::sort(v.begin(), v.end());
            const int i_low  = int(low  * v.size());
            const int i_high = int(high * v.size());
            return std::make_pair(v[i_low], v[i_high]);
        };

        std::tie(q_min,  q_max ) = ranged(all_q ,  0.0025, 0.9925);
        std::tie(vs_min, vs_max) = ranged(all_vs, 0.0025, 0.9925);
        std::tie(rs_min, rs_max) = ranged(all_rs, 0.0025, 0.9925);

        q_dx  = (q_max  - q_min ) / double(n_q  - 1);
        vs_dx = (vs_max - vs_min) / double(n_vs - 1);
        rs_dx = (rs_max - rs_min) / double(n_rs - 1);

        std::cout << DIM << std::fixed << std::setprecision(3)
                  << "   q  : [" << q_min  << ", " << q_max  << "] × " << n_q  << '\n'
                  << "   v_s: [" << std::setprecision(1) << vs_min << ", " << vs_max << "] km/s × "
                  << n_vs << '\n'
                  << "   r_s: [" << std::setprecision(4) << rs_min << ", " << rs_max << "] × "
                  << n_rs << RESET << "\n";

        /* ─── allocate axes ─────────────────────────────────────────────────────── */
        q_grid .resize(n_q );
        vs_grid.resize(n_vs);
        rs_grid.resize(n_rs);
        for (int i = 0; i < n_q ; ++i) q_grid [i] = q_min  + i * q_dx;
        for (int i = 0; i < n_vs; ++i) vs_grid[i] = vs_min + i * vs_dx;
        for (int i = 0; i < n_rs; ++i) rs_grid[i] = rs_min + i * rs_dx;

        /* ─── allocate pdf arrays and helper lambda for one strip ──────────────── */
        pdf_grid_q .assign(n_incl, std::vector<double>(n_q ,  0.0));
        pdf_grid_vs.assign(n_incl, std::vector<double>(n_vs, 0.0));
        pdf_grid_rs.assign(n_incl, std::vector<double>(n_rs, 0.0));

        const double inv_sqrt_2pi = 1.0 / std::sqrt(2.0 * M_PI);

        auto regenerate_strip = [&](int i_incl)
        {
            const double incl  = inclination_grid[i_incl];
            const double sin_i = std::sin(incl * M_PI / 180.0);

            std::vector<double> ratios, v_scales, r_scales;
            ratios .reserve(nsamp);
            v_scales.reserve(nsamp);
            r_scales.reserve(nsamp);

            for (int s = 0; s < nsamp; ++s) {
                const double m1 = sample_pos(dist_m1);
                const double m2 = sample_pos(dist_m2);
                const double K  = sample_pos(dist_K );
                const double R  = sample_pos(dist_R );
                const double P  = sample_pos(dist_P );

                const double q  = mass_ratio_from_inclination(incl, m1, m2);
                const double v_s = (1.0 + 1.0 / q) * K / sin_i;
                const double r_s = 2.0 * M_PI * R /
                                   (P * day_to_sec * v_s * km_to_solrad);

                if (q  <= 0.0 || v_s <= 0.0 || r_s <= 0.0 ||
                    !std::isfinite(q) || !std::isfinite(v_s) || !std::isfinite(r_s))
                    continue;

                ratios .push_back(q );
                v_scales.push_back(v_s);
                r_scales.push_back(r_s);
            }

            const std::size_t n_i = ratios.size();
            if (n_i == 0) {
                std::fill(pdf_grid_q [i_incl].begin(), pdf_grid_q [i_incl].end(), 1e-12);
                std::fill(pdf_grid_vs[i_incl].begin(), pdf_grid_vs[i_incl].end(), 1e-12);
                std::fill(pdf_grid_rs[i_incl].begin(), pdf_grid_rs[i_incl].end(), 1e-12);
                return;
            }

            const double h_q  = scott_bandwidth(ratios);
            const double h_vs = scott_bandwidth(v_scales);
            const double h_rs = scott_bandwidth(r_scales);

            for (int iq = 0; iq < n_q; ++iq) {
                const double x = q_grid[iq];
                double sum = 0.0;
                for (double xi : ratios) {
                    const double u = (x - xi) / h_q;
                    sum += std::exp(-0.5 * u * u);
                }
                pdf_grid_q[i_incl][iq] = sum * inv_sqrt_2pi / (n_i * h_q);
            }
            for (int iv = 0; iv < n_vs; ++iv) {
                const double x = vs_grid[iv];
                double sum = 0.0;
                for (double xi : v_scales) {
                    const double u = (x - xi) / h_vs;
                    sum += std::exp(-0.5 * u * u);
                }
                pdf_grid_vs[i_incl][iv] = sum * inv_sqrt_2pi / (n_i * h_vs);
            }
            for (int ir = 0; ir < n_rs; ++ir) {
                const double x = rs_grid[ir];
                double sum = 0.0;
                for (double xi : r_scales) {
                    const double u = (x - xi) / h_rs;
                    sum += std::exp(-0.5 * u * u);
                }
                pdf_grid_rs[i_incl][ir] = sum * inv_sqrt_2pi / (n_i * h_rs);
            }
        };

        /* ─────────── first pass over all inclination strips ───────────────────── */
        for (int i = 0; i < n_incl; ++i) {
            if (i > 0 && i % std::max(1, n_incl / 5) == 0) {
                std::cout << "\r" << BRIGHT_CYAN << "   Progress: "
                          << (100 * i / n_incl) << "% (" << i << "/" << n_incl << ")" << RESET << std::flush;
            }
            if (std::sin(inclination_grid[i] * M_PI / 180.0) < 1e-6) continue;
            regenerate_strip(i);
        }

        /* ─────────── outlier detection & clean-up ─────────────────────────────── */
        std::vector<double> div_q, div_vs, div_rs;
        compute_divergence(pdf_grid_q , div_q );
        compute_divergence(pdf_grid_vs, div_vs);
        compute_divergence(pdf_grid_rs, div_rs);

        auto threshold = [](const std::vector<double>& v) {
            double mean = 0.0, var = 0.0; int n = 0;
            for (double x : v) if (x > 0.0) { mean += x; ++n; }
            mean /= n;
            for (double x : v) if (x > 0.0) var += (x - mean) * (x - mean);
            var /= n;
            return mean + 3.0 * std::sqrt(var);
        };

        const double thr_q  = threshold(div_q );
        const double thr_vs = threshold(div_vs);
        const double thr_rs = threshold(div_rs);

        std::vector<int> outliers;
        for (int i = 1; i < n_incl - 1; ++i) {
            if (div_q [i] > thr_q  || div_vs[i] > thr_vs ||
                div_rs[i] > thr_rs)
                outliers.push_back(i);
        }

        if (!outliers.empty()) {
            std::cout << "\n   Refinement: " << outliers.size()
                      << " divergent inclination strip(s) → resampling.\n";
            for (int idx : outliers) regenerate_strip(idx);

            /* recompute divergences to show improvement */
            compute_divergence(pdf_grid_q , div_q );
            compute_divergence(pdf_grid_vs, div_vs);
            compute_divergence(pdf_grid_rs, div_rs);

            int stubborn = 0;
            for (int i : outliers)
                if (div_q[i]  > thr_q ||
                    div_vs[i] > thr_vs ||
                    div_rs[i] > thr_rs) ++stubborn;

            if (stubborn == 0)
                std::cout << "   Refinement successful – all strips now within 3σ.\n";
            else
                std::cout << "   Warning: " << stubborn
                          << " strip(s) still above 3σ after regeneration.\n";
        }

        std::cout << BRIGHT_GREEN << "✓ Grid pre-computation complete!" << RESET << "\n";
    }

    /* ───────── fast bilinear interpolation for q / v_s / r_s ─────────────────── */
    double evaluate_q (double incl, double q ) const { return interp(pdf_grid_q , incl, q , incl_min, incl_dx, n_incl, q_min , q_dx , n_q ); }
    double evaluate_vs(double incl, double vs) const { return interp(pdf_grid_vs, incl, vs, incl_min, incl_dx, n_incl, vs_min, vs_dx, n_vs); }
    double evaluate_rs(double incl, double rs) const { return interp(pdf_grid_rs, incl, rs, incl_min, incl_dx, n_incl, rs_min, rs_dx, n_rs); }

    /* combined log-pdf ----------------------------------------------------------- */
    double log_combined_pdf(double incl, double q, double vs, double rs) const
    {
        return std::log(evaluate_q(incl, q))  +
               std::log(evaluate_vs(incl, vs))+
               std::log(evaluate_rs(incl, rs));
    }

    /* for diagnostics ------------------------------------------------------------ */
    bool in_bounds(double i, double q, double vs, double rs) const
    {
        return (i  >= incl_min && i  <= incl_max &&
                q  >= q_min   && q  <= q_max   &&
                vs >= vs_min  && vs <= vs_max  &&
                rs >= rs_min  && rs <= rs_max );
    }

    void print_info() const
    {
        std::cout << "Grid info:\n"
                  << "  Inclination: [" << incl_min << ", " << incl_max << "] deg, "
                  << n_incl << " points\n"
                  << "  q          : [" << q_min    << ", " << q_max    << "], "
                  << n_q << " points\n"
                  << "  v_s (km/s) : [" << vs_min   << ", " << vs_max   << "], "
                  << n_vs << " points\n"
                  << "  r_s        : [" << rs_min   << ", " << rs_max   << "], "
                  << n_rs << " points\n";
    }

private:
    /* ───── generic bilinear interpolator (private) ───────────────────────────── */
    static double interp(const std::vector<std::vector<double>>& grid,
                         double x, double y,
                         double x0, double dx, int nx,
                         double y0, double dy, int ny)
    {
        if (x < x0 || x > x0 + dx * (nx - 1) ||
            y < y0 || y > y0 + dy * (ny - 1))
            return 1e-12;

        double px = (x - x0) / dx;
        double py = (y - y0) / dy;
        int ix = int(px);
        int iy = int(py);
        if (ix >= nx - 1) ix = nx - 2;
        if (iy >= ny - 1) iy = ny - 2;
        double fx = px - ix;
        double fy = py - iy;

        const double v00 = grid[ix    ][iy    ];
        const double v10 = grid[ix + 1][iy    ];
        const double v01 = grid[ix    ][iy + 1];
        const double v11 = grid[ix + 1][iy + 1];

        const double v0 = v00 * (1.0 - fx) + v10 * fx;
        const double v1 = v01 * (1.0 - fx) + v11 * fx;
        const double v  = v0  * (1.0 - fy) + v1  * fy;
        return std::max(v, 1e-12);
    }
};

/* ─────────────────────── global interface (unchanged) ───────────────────────── */
void initialize_mass_ratio_pdf_grid(
    double m1_mean, double m1_err,
    double m2_mean, double m2_err,
    double K_mean,  double K_err,
    double R_mean,  double R_err,
    double P_mean,  double P_err,
    double incl_min, double incl_max, int n_incl,
    int n_q, int n_vs, int n_rs, int nsamp)
{
    delete g_mass_ratio_grid;
    g_mass_ratio_grid = new MassRatioPDFGrid(
        m1_mean, m1_err, m2_mean, m2_err,
        K_mean, K_err, R_mean, R_err, P_mean, P_err,
        incl_min, incl_max, n_incl,
        n_q, n_vs, n_rs, nsamp);
}

double log_mass_ratio_pdf(double i, double q, double vs, double rs)
{
    if (!g_mass_ratio_grid)
        throw std::runtime_error("PDF grid not initialised.");
    return g_mass_ratio_grid->log_combined_pdf(i, q, vs, rs);
}

double mass_ratio_pdf    (double i, double q ) { return g_mass_ratio_grid->evaluate_q (i, q ); }
double velocity_scale_pdf(double i, double vs) { return g_mass_ratio_grid->evaluate_vs(i, vs); }
double radius_scale_pdf  (double i, double rs) { return g_mass_ratio_grid->evaluate_rs(i, rs); }

void cleanup_mass_ratio_pdf_grid() { delete g_mass_ratio_grid; g_mass_ratio_grid = nullptr; }
void print_grid_info()             { if (g_mass_ratio_grid) g_mass_ratio_grid->print_info(); }

bool check_in_bounds(double i, double q, double vs, double rs)
{
    return g_mass_ratio_grid && g_mass_ratio_grid->in_bounds(i, q, vs, rs);
}