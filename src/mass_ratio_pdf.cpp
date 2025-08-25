// mass_ratio_pdf.cpp
/*****************************************************************************************
 Fast   p(q | i) · p(v_s | i) · p(r_s | i)   grid
 ─────────────────────────────────────────────────────────────────────────────────────────
 • 2-D (inclination, parameter) kernel density estimate
   – eliminates the former “banding / divergent strips”.
 • Binned KDE  (= Gaussian blur of a 2-D histogram),
   therefore O(Ngrid · σ) and very cache friendly.
 • Public interface unchanged – simply replace the old file and recompile.
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

#ifdef _OPENMP
#   include <omp.h>
#endif

#include "mass_ratio_pdf.h"
#include "lcurve_base/lcurve.h"

/* ─────────────────────────── constants ──────────────────────────── */
static constexpr double day_to_sec   = 86400.0;
static constexpr double km_to_solrad = 1.0 / 695700.0;

/* ─────────────────────────── globals  ───────────────────────────── */
class MassRatioPDFGrid;   // forward
MassRatioPDFGrid* g_mass_ratio_grid = nullptr;

/* ───────────────────────── helpers ──────────────────────────────── */
double bisect(const std::function<double(double)>& f,
              double c_lo, double c_hi,
              double tol = 1e-9, int max_iter = 100)
{
    double f_lo = f(c_lo), f_hi = f(c_hi);
    if (f_lo * f_hi > 0.0)
        throw std::runtime_error("Bisection end points have same sign.");

    double c_mid = 0.0, f_mid;
    for (int i = 0; i < max_iter; ++i) {
        c_mid = 0.5 * (c_lo + c_hi);
        f_mid = f(c_mid);
        if (std::fabs(f_mid) < tol || 0.5 * (c_hi - c_lo) < tol) return c_mid;
        if (f_lo * f_mid <= 0.0) { c_hi = c_mid; f_hi = f_mid; }
        else                     { c_lo = c_mid; f_lo = f_mid; }
    }
    return c_mid;   // may not be fully converged
}

/* invert  m sin³i  →  q ----------------------------------------------------------- */
double mass_ratio_from_inclination(double inclination_deg,
                                   double mass1, double min_mass2)
{
    const double A      = std::pow(min_mass2, 3) / std::pow(min_mass2 + mass1, 2);
    const double target = std::pow(std::sin(inclination_deg * M_PI / 180.0), 3);

    auto f = [&](double c) { return A * std::pow(c + mass1, 2) / std::pow(c, 3) - target; };
    return bisect(f, 1e-6, 1e5) / mass1;
}

/* Scott / Silverman rule of thumb for multivariate (here: d = 2) */
inline double scott_bw_2d(double sigma, std::size_t n_samples)
{
    if (n_samples < 2) return sigma;
    const double n    = static_cast<double>(n_samples);
    const double fac  = std::pow(n, -1.0 / 6.0);   // d = 2  → 1/(d+4)
    return 1.06 * sigma * fac;
}

/* ────────────────────── Gaussian kernel helper ─────────────────── */
static std::vector<double> gaussian_kernel(double sigma_grid_units)
{
    const int radius = std::max(1, int(std::ceil(3.0 * sigma_grid_units)));
    const int size   = 2 * radius + 1;
    std::vector<double> k(size);
    const double inv_2s2 = 1.0 / (2.0 * sigma_grid_units * sigma_grid_units);
    double norm = 0.0;
    for (int dx = -radius; dx <= radius; ++dx) {
        const double w = std::exp(-dx * dx * inv_2s2);
        k[dx + radius] = w;
        norm          += w;
    }
    for (double &v : k) v /= norm;          // normalise
    return k;
}

/* ────────────────────── 2-D Gaussian blur (separable) ──────────── */
template<typename Matrix>
static void gaussian_blur(Matrix       &m,         // in-place
                          double        sigma_x,   // in grid index units
                          double        sigma_y)
{
    const int nx = (int)m.size();
    const int ny = (int)m.front().size();
    const auto kx = gaussian_kernel(sigma_x);
    const auto ky = gaussian_kernel(sigma_y);
    const int rx = int(kx.size() / 2);
    const int ry = int(ky.size() / 2);

    /* ---- blur along X (inclination) ---- */
    Matrix tmp(nx, std::vector<double>(ny, 0.0));

    #pragma omp parallel for schedule(static)
    for (int ix = 0; ix < nx; ++ix)
        for (int iy = 0; iy < ny; ++iy) {
            double acc = 0.0;
            for (int dx = -rx; dx <= rx; ++dx) {
                const int jx = ix + dx;
                if (jx < 0 || jx >= nx) continue;
                acc += m[jx][iy] * kx[dx + rx];
            }
            tmp[ix][iy] = acc;
        }

    /* ---- blur along Y (parameter) ---- */
    #pragma omp parallel for schedule(static)
    for (int ix = 0; ix < nx; ++ix)
        for (int iy = 0; iy < ny; ++iy) {
            double acc = 0.0;
            for (int dy = -ry; dy <= ry; ++dy) {
                const int jy = iy + dy;
                if (jy < 0 || jy >= ny) continue;
                acc += tmp[ix][jy] * ky[dy + ry];
            }
            m[ix][iy] = acc;
        }
}

/* ────────────────────────  Class definition  ───────────────────── */
class MassRatioPDFGrid
{
    /* axes */
    std::vector<double> inclination_grid, q_grid, vs_grid, rs_grid;
    double incl_min, incl_max, incl_dx;
    double q_min,    q_max,    q_dx;
    double vs_min,   vs_max,   vs_dx;
    double rs_min,   rs_max,   rs_dx;
    int    n_incl, n_q, n_vs, n_rs;

    /* PDFs on (inclination, parameter) grid */
    std::vector<std::vector<double>> pdf_q, pdf_vs, pdf_rs;

    /* Monte-Carlo samples, used only during construction */
    struct StripSamples { std::vector<double> q, vs, rs; };
    std::vector<StripSamples> strips;

public:
    MassRatioPDFGrid(
        double m1_mu,  double m1_sig,
        double m2_mu,  double m2_sig,
        double K_mu,   double K_sig,
        double R_mu,   double R_sig,
        double P_mu,   double P_sig,
        double i_min,  double i_max, int n_i,
        int n_q_, int n_vs_, int n_rs_, int nsamp_tot)
    : incl_min(i_min), incl_max(i_max), n_incl(n_i),
      n_q(n_q_), n_vs(n_vs_), n_rs(n_rs_),
      strips(n_incl)
    {
        /* -------------------- axis initialisation -------------------- */
        incl_dx = (incl_max - incl_min) / double(n_incl - 1);
        inclination_grid.resize(n_incl);
        for (int i = 0; i < n_incl; ++i) inclination_grid[i] = incl_min + i * incl_dx;

        /* -------------------- Monte-Carlo draw ----------------------- */
        const auto make_dist = [](double mu, double s)
                               { return std::normal_distribution<double>(mu, s); };

        #pragma omp parallel for schedule(dynamic)
        for (int idx = 0; idx < n_incl; ++idx)
        {
            const double inc_deg = inclination_grid[idx];
            const double sin_i   = std::sin(inc_deg * M_PI / 180.0);
            if (sin_i < 1e-6) continue;

            std::mt19937_64 gen( std::random_device{}() + idx );
            auto    d_m1 = make_dist(m1_mu, m1_sig);
            auto    d_m2 = make_dist(m2_mu, m2_sig);
            auto    d_K  = make_dist(K_mu , K_sig );
            auto    d_R  = make_dist(R_mu , R_sig );
            auto    d_P  = make_dist(P_mu , P_sig );

            const int nsamp_strip = nsamp_tot / n_incl;

            auto draw_pos = [&](auto &dist)
            {
                while (true) { double v = dist(gen); if (v > 0.0 && std::isfinite(v)) return v; }
            };

            auto &S = strips[idx];
            S.q .reserve(nsamp_strip);
            S.vs.reserve(nsamp_strip);
            S.rs.reserve(nsamp_strip);

            for (int s = 0; s < nsamp_strip; ++s)
            {
                const double m1 = draw_pos(d_m1);
                const double m2 = draw_pos(d_m2);
                const double K  = draw_pos(d_K );
                const double R  = draw_pos(d_R );
                const double P  = draw_pos(d_P );

                const double q  = mass_ratio_from_inclination(inc_deg, m1, m2);
                if (q <= 0.0 || !std::isfinite(q)) continue;

                const double vs = (1.0 + 1.0 / q) * K / sin_i;
                const double rs = 2.0 * M_PI * R /
                                  (P * day_to_sec * vs * km_to_solrad);

                if (vs <= 0.0 || rs <= 0.0 || !std::isfinite(vs) || !std::isfinite(rs))
                    continue;

                S.q .push_back(q );
                S.vs.push_back(vs);
                S.rs.push_back(rs);
            }
        }

        /* ---------------- compute global parameter ranges ------------- */
        std::vector<double> all_q, all_vs, all_rs;
        for (auto &s : strips) {
            all_q .insert(all_q .end(), s.q .begin(), s.q .end());
            all_vs.insert(all_vs.end(), s.vs.begin(), s.vs.end());
            all_rs.insert(all_rs.end(), s.rs.begin(), s.rs.end());
        }
        if (all_q.empty()) throw std::runtime_error("No valid MC samples.");

        const auto percent = [](std::vector<double> &v, double f)
        {
            const std::size_t k = std::size_t(f * (v.size() - 1));
            std::nth_element(v.begin(), v.begin() + k, v.end());
            return v[k];
        };

        q_min  = percent(all_q , 0.0025);  q_max  = percent(all_q , 0.9975);
        vs_min = percent(all_vs, 0.0025);  vs_max = percent(all_vs, 0.9975);
        rs_min = percent(all_rs, 0.0025);  rs_max = percent(all_rs, 0.9975);

        q_dx  = (q_max  - q_min ) / double(n_q  - 1);
        vs_dx = (vs_max - vs_min) / double(n_vs - 1);
        rs_dx = (rs_max - rs_min) / double(n_rs - 1);

        q_grid .resize(n_q ); for (int i = 0; i < n_q ; ++i) q_grid [i] = q_min  + i * q_dx;
        vs_grid.resize(n_vs); for (int i = 0; i < n_vs; ++i) vs_grid[i] = vs_min + i * vs_dx;
        rs_grid.resize(n_rs); for (int i = 0; i < n_rs; ++i) rs_grid[i] = rs_min + i * rs_dx;

        /* ----------- histograms (inclination, parameter) -------------- */
        std::vector<std::vector<double>> Hq (n_incl, std::vector<double>(n_q , 0.0));
        std::vector<std::vector<double>> Hvs(n_incl, std::vector<double>(n_vs, 0.0));
        std::vector<std::vector<double>> Hrs(n_incl, std::vector<double>(n_rs, 0.0));

        auto bin_index = [](double x, double x0, double dx, int nx) -> int
        {
            const int idx = int(std::floor((x - x0) / dx + 0.5));
            return (idx < 0 || idx >= nx) ? -1 : idx;
        };

        for (int i = 0; i < n_incl; ++i)
        {
            const auto &S = strips[i];
            for (double v : S.q) {
                int k = bin_index(v, q_min, q_dx, n_q);
                if (k >= 0) Hq[i][k] += 1.0;
            }
            for (double v : S.vs) {
                int k = bin_index(v, vs_min, vs_dx, n_vs);
                if (k >= 0) Hvs[i][k] += 1.0;
            }
            for (double v : S.rs) {
                int k = bin_index(v, rs_min, rs_dx, n_rs);
                if (k >= 0) Hrs[i][k] += 1.0;
            }
        }


        /* --------- row-wise normalisation:  ∑ p(x|i) Δx = 1 ---------- */
        pdf_q  = Hq;  pdf_vs = Hvs; pdf_rs = Hrs;

        const auto norm_rows = [&](auto &M, double dx)
        {
            #pragma omp parallel for schedule(static)
            for (int ix = 0; ix < n_incl; ++ix)
            {
                double row_sum = std::accumulate(M[ix].begin(), M[ix].end(), 0.0);
                if (row_sum <= 0.0) {
                    for (double &v : M[ix]) v = 1e-12;
                    continue;
                }
                const double norm = row_sum * dx;
                for (double &v : M[ix]) v = std::max(v / norm, 1e-12);
            }
        };

        norm_rows(pdf_q , q_dx );
        norm_rows(pdf_vs, vs_dx);
        norm_rows(pdf_rs, rs_dx);
    }

    /* ─────────── bilinear interpolation helper ─────────── */
    template<typename Grid>
    inline double interp(const Grid &G,
                         double i_deg, double x,
                         double x0,  double dx,  int nx)      const
    {
        if (i_deg < incl_min || i_deg > incl_max ||
            x      < x0       || x      > x0 + dx * (nx - 1))
            return 1e-12;

        const double pi = (i_deg - incl_min) / incl_dx;
        int   ii = int(pi);  if (ii >= n_incl - 1) ii = n_incl - 2;
        const double fi = pi - ii;

        const double pj = (x - x0) / dx;
        int   jj = int(pj);  if (jj >= nx - 1) jj = nx - 2;
        const double fj = pj - jj;

        const double v00 = G[ii    ][jj    ];
        const double v10 = G[ii + 1][jj    ];
        const double v01 = G[ii    ][jj + 1];
        const double v11 = G[ii + 1][jj + 1];

        const double v0 = v00 * (1.0 - fi) + v10 * fi;
        const double v1 = v01 * (1.0 - fi) + v11 * fi;
        const double v  = v0  * (1.0 - fj) + v1  * fj;

        return std::max(v, 1e-12);
    }

    /* ─────────── public evaluators ─────────── */
    double pdf_q_i (double i, double q ) const { return interp(pdf_q , i, q , q_min , q_dx , n_q ); }
    double pdf_vs_i(double i, double vs) const { return interp(pdf_vs, i, vs, vs_min, vs_dx, n_vs); }
    double pdf_rs_i(double i, double rs) const { return interp(pdf_rs, i, rs, rs_min, rs_dx, n_rs); }

    double log_combined(double i, double q, double vs, double rs) const
    {
        return std::log(pdf_q_i(i,q)) + std::log(pdf_vs_i(i,vs))
                                     + std::log(pdf_rs_i(i,rs));
    }

    bool in_bounds(double i, double q, double vs, double rs) const
    {
        return i  >= incl_min && i  <= incl_max &&
               q  >= q_min   && q  <= q_max   &&
               vs >= vs_min  && vs <= vs_max  &&
               rs >= rs_min  && rs <= rs_max ;
    }

    void info() const
    {
        std::cout << "PDF grid:\n"
                  << "  i   : [" << incl_min << ", " << incl_max << "]°, "
                  << n_incl << " nodes\n"
                  << "  q   : [" << q_min    << ", " << q_max    << "], "
                  << n_q    << " nodes\n"
                  << "  v_s : [" << vs_min   << ", " << vs_max   << "] km/s, "
                  << n_vs   << " nodes\n"
                  << "  r_s : [" << rs_min   << ", " << rs_max   << "], "
                  << n_rs   << " nodes\n";
    }
};

/* ─────────────────────── global façade (unchanged) ────────────────────── */
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
        K_mean,  K_err,  R_mean,  R_err,
        P_mean,  P_err,
        incl_min, incl_max, n_incl,
        n_q, n_vs, n_rs, nsamp);
}

double log_mass_ratio_pdf(double i, double q, double vs, double rs)
{
    if (!g_mass_ratio_grid)
        throw std::runtime_error("PDF grid not initialised.");
    return g_mass_ratio_grid->log_combined(i, q, vs, rs);
}

double mass_ratio_pdf    (double i, double q ) { return g_mass_ratio_grid->pdf_q_i (i,q ); }
double velocity_scale_pdf(double i, double vs) { return g_mass_ratio_grid->pdf_vs_i(i,vs); }
double radius_scale_pdf  (double i, double rs) { return g_mass_ratio_grid->pdf_rs_i(i,rs); }

void cleanup_mass_ratio_pdf_grid() { delete g_mass_ratio_grid; g_mass_ratio_grid = nullptr; }
void print_grid_info()             { if (g_mass_ratio_grid) g_mass_ratio_grid->info(); }

bool check_in_bounds(double i, double q, double vs, double rs)
{
    return g_mass_ratio_grid && g_mass_ratio_grid->in_bounds(i, q, vs, rs);
}