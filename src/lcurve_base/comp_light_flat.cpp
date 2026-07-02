/* Fast structure-of-arrays flux summation kernels.
 *
 * These reproduce the per-element arithmetic of comp_light.cpp exactly
 * (same operation order within each element); only the order in which
 * element contributions are added to the total differs (partitioned
 * grids, SIMD reductions), which perturbs sums at the rounding level.
 */
#include <cmath>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../new_subs.h"
#include "constants.h"
#include "../lroche_base/roche.h"
#include "lcurve.h"

using std::vector;

/* ---------------- FlatGrid::build ---------------------------- */

void Lcurve::FlatGrid::build(const vector<Point> &pts) {
    n = pts.size();
    size_t c0 = 0, c1 = 0;
    for (const auto &p : pts) {
        size_t k = p.eclipse.size();
        c0 += (k == 0);
        c1 += (k == 1);
    }
    n0 = c0;
    n1 = c0 + c1;

    dx.resize(n); dy.resize(n); dz.resize(n);
    px.resize(n); py.resize(n); pz.resize(n);
    flux.resize(n);
    in1.resize(n1 - n0); out1.resize(n1 - n0);
    moff.clear(); min_.clear(); mout_.clear();
    moff.reserve(n - n1 + 1);
    moff.push_back(0);

    size_t i0 = 0, i1 = n0, i2 = n1;
    for (const auto &p : pts) {
        size_t k = p.eclipse.size();
        size_t i = (k == 0) ? i0++ : (k == 1 ? i1++ : i2++);
        dx[i] = p.dirn.x(); dy[i] = p.dirn.y(); dz[i] = p.dirn.z();
        px[i] = p.posn.x(); py[i] = p.posn.y(); pz[i] = p.posn.z();
        flux[i] = p.flux;
        if (k == 1) {
            in1[i - n0] = p.eclipse[0].first;
            out1[i - n0] = p.eclipse[0].second;
        } else if (k >= 2) {
            for (const auto &e : p.eclipse) {
                min_.push_back(e.first);
                mout_.push_back(e.second);
            }
            moff.push_back(static_cast<int>(min_.size()));
        }
    }
}

/* ---------------- element kernels ----------------------------- */

namespace {

using Lcurve::FlatGrid;
using Lcurve::LDC;
using Lcurve::Point;

//! limb-darkening coefficients unpacked for the kernels
struct LdcC {
    double c1, c2, c3, c4, base, mucrit;
};

//! branchless I(mu); LTYPE 0 = POLY, 1 = CLARET
template<int LTYPE>
inline double imu_bl(double mu, const LdcC &L) {
    double m = std::min(mu, 1.0);
    double im;
    if constexpr (LTYPE == 0) {
        double om = 1.0 - m;
        im = 1.0 - om * (L.c1 + om * (L.c2 + om * (L.c3 + om * L.c4)));
    } else {
        double msq = std::sqrt(std::max(m, 0.0));
        im = L.base + msq * (L.c1 + msq * (L.c2 + msq * (L.c3 + msq * L.c4)));
    }
    return mu > 0.0 ? im : 0.0;
}

//! raw-pointer view of a FlatGrid, so the SIMD loops see plain arrays
struct GridPtrs {
    const double *__restrict dx, *__restrict dy, *__restrict dz;
    const double *__restrict px, *__restrict py, *__restrict pz;
    const double *__restrict flux;
    const double *__restrict in1, *__restrict out1;
    explicit GridPtrs(const FlatGrid &g)
        : dx(g.dx.data()), dy(g.dy.data()), dz(g.dz.data()),
          px(g.px.data()), py(g.py.data()), pz(g.pz.data()),
          flux(g.flux.data()), in1(g.in1.data()), out1(g.out1.data()) {}
};

//! star-1 element flux (no visibility handling; caller masks eclipses)
template<int LTYPE, bool BEAM>
inline double s1_elem(const GridPtrs &g, size_t i,
                      double ex, double ey, double ez, const LdcC &L,
                      double beam, double spin, double VFAC, double XCOFM) {
    double mu = ex * g.dx[i] + ey * g.dy[i] + ez * g.dz[i];
    double val;
    if constexpr (!BEAM) {
        val = mu * g.flux[i] * imu_bl<LTYPE>(mu, L);
    } else {
        double vx = -VFAC * spin * g.py[i];
        double vy = VFAC * (spin * g.px[i] - XCOFM);
        double vr = -(ex * vx + ey * vy);
        double vn = g.dx[i] * vx + g.dy[i] * vy;
        double mud = mu - mu * vr - vn;
        val = mu * g.flux[i] * (1.0 - beam * vr) * imu_bl<LTYPE>(mud, L);
    }
    return mu > L.mucrit ? val : 0.0;
}

//! star-2 element flux
template<int LTYPE, bool BEAM, bool GLENS>
inline double s2_elem(const GridPtrs &g, size_t i,
                      double ex, double ey, double ez, const LdcC &L,
                      double beam, double spin, double VFAC, double XCOFM,
                      double rlens1) {
    double mu = ex * g.dx[i] + ey * g.dy[i] + ez * g.dz[i];

    double magn = 1.0;
    if constexpr (GLENS) {
        double sx = g.px[i], sy = g.py[i], sz = g.pz[i];
        double d = -(sx * ex + sy * ey + sz * ez);
        double qx = sx + d * ex, qy = sy + d * ey, qz = sz + d * ez;
        double p = std::sqrt(qx * qx + qy * qy + qz * qz);
        double ph = 0.5 * p;
        double rd = rlens1 * d;
        double pd = (ph * ph > 25.0 * rd) ? p + rd / p
                                          : ph + std::sqrt(ph * ph + rd);
        double mg = pd * pd / (pd - ph) / ph / 4.0;
        magn = d > 0.0 ? mg : 1.0;
    }

    double val;
    if constexpr (!BEAM) {
        val = mu * magn * g.flux[i] * imu_bl<LTYPE>(mu, L);
    } else {
        double vx = -VFAC * spin * g.py[i];
        double vy = VFAC * (spin * (g.px[i] - 1.0) + 1.0 - XCOFM);
        double vr = -(ex * vx + ey * vy);
        double vn = g.dx[i] * vx + g.dy[i] * vy;
        double mud = mu - mu * vr - vn;
        val = mu * magn * g.flux[i] * (1.0 - beam * vr) * imu_bl<LTYPE>(mud, L);
    }
    return mu > L.mucrit ? val : 0.0;
}

//! same eclipse test as Point::visible for one range
inline bool ecl_range(double phin, double in, double out) {
    return (phin >= in && phin <= out) || phin <= out - 1.0;
}

template<int LTYPE, bool BEAM>
double sum_star1_t(const FlatGrid &fg, double ex, double ey, double ez,
                   double phin, const LdcC &L, double beam, double spin,
                   double VFAC, double XCOFM) {
    const GridPtrs g(fg);
    const size_t n0 = fg.n0, n1 = fg.n1, n = fg.n;
    double s = 0.0;
    #pragma omp simd reduction(+:s)
    for (size_t i = 0; i < n0; i++)
        s += s1_elem<LTYPE, BEAM>(g, i, ex, ey, ez, L, beam, spin, VFAC, XCOFM);

    #pragma omp simd reduction(+:s)
    for (size_t i = n0; i < n1; i++) {
        double v = s1_elem<LTYPE, BEAM>(g, i, ex, ey, ez, L, beam, spin,
                                        VFAC, XCOFM);
        s += ecl_range(phin, g.in1[i - n0], g.out1[i - n0]) ? 0.0 : v;
    }

    for (size_t i = n1; i < n; i++) {
        bool ecl = false;
        for (int j = fg.moff[i - n1]; j < fg.moff[i - n1 + 1]; j++)
            if (ecl_range(phin, fg.min_[j], fg.mout_[j])) { ecl = true; break; }
        if (!ecl)
            s += s1_elem<LTYPE, BEAM>(g, i, ex, ey, ez, L, beam, spin,
                                      VFAC, XCOFM);
    }
    return s;
}

template<int LTYPE, bool BEAM, bool GLENS>
double sum_star2_t(const FlatGrid &fg, double ex, double ey, double ez,
                   double phin, const LdcC &L, double beam, double spin,
                   double VFAC, double XCOFM, double rlens1) {
    const GridPtrs g(fg);
    const size_t n0 = fg.n0, n1 = fg.n1, n = fg.n;
    double s = 0.0;
    #pragma omp simd reduction(+:s)
    for (size_t i = 0; i < n0; i++)
        s += s2_elem<LTYPE, BEAM, GLENS>(g, i, ex, ey, ez, L, beam, spin,
                                         VFAC, XCOFM, rlens1);

    #pragma omp simd reduction(+:s)
    for (size_t i = n0; i < n1; i++) {
        double v = s2_elem<LTYPE, BEAM, GLENS>(g, i, ex, ey, ez, L, beam,
                                               spin, VFAC, XCOFM, rlens1);
        s += ecl_range(phin, g.in1[i - n0], g.out1[i - n0]) ? 0.0 : v;
    }

    for (size_t i = n1; i < n; i++) {
        bool ecl = false;
        for (int j = fg.moff[i - n1]; j < fg.moff[i - n1 + 1]; j++)
            if (ecl_range(phin, fg.min_[j], fg.mout_[j])) { ecl = true; break; }
        if (!ecl)
            s += s2_elem<LTYPE, BEAM, GLENS>(g, i, ex, ey, ez, L, beam, spin,
                                             VFAC, XCOFM, rlens1);
    }
    return s;
}

/* ---- multi-phase blocked sweeps ------------------------------ */

// Faces per block; 6 arrays x 8 B x 1024 = ~48 kB, L1/L2-resident. Small
// enough that a 16-thread team gets ~80 blocks from a 250-lat grid, which
// balances well.
constexpr size_t FACE_BLOCK = 1024;

template<int LTYPE, bool BEAM>
void sum_star1_multi_t(const FlatGrid &fg, const Lcurve::PhaseBatch &pb,
                       const LdcC &L, double beam, double spin,
                       double VFAC, double XCOFM, double *out) {
    const GridPtrs g(fg);
    const size_t n0 = fg.n0, n1 = fg.n1, n = fg.n;
    const int nent = static_cast<int>(pb.size());
    const long nblk = static_cast<long>((n + FACE_BLOCK - 1) / FACE_BLOCK);
    std::fill(out, out + nent, 0.0);

    #pragma omp parallel if(!omp_in_parallel())
    {
        std::vector<double> acc(nent, 0.0);
        #pragma omp for schedule(dynamic) nowait
        for (long b = 0; b < nblk; b++) {
            const size_t lo = b * FACE_BLOCK;
            const size_t hi = std::min(n, lo + FACE_BLOCK);
            const size_t a1 = std::min(hi, n0);            // plain segment
            const size_t b0 = std::max(lo, n0);            // 1-range segment
            const size_t b1 = std::min(hi, n1);
            const size_t c0 = std::max(lo, n1);            // ragged segment

            for (int k = 0; k < nent; k++) {
                const double ex = pb.ex[k], ey = pb.ey[k], ez = pb.ez[k];
                const double phin = pb.phin[k];
                double s = 0.0;

                #pragma omp simd reduction(+:s)
                for (size_t i = lo; i < a1; i++)
                    s += s1_elem<LTYPE, BEAM>(g, i, ex, ey, ez, L, beam,
                                              spin, VFAC, XCOFM);

                #pragma omp simd reduction(+:s)
                for (size_t i = b0; i < b1; i++) {
                    double v = s1_elem<LTYPE, BEAM>(g, i, ex, ey, ez, L,
                                                    beam, spin, VFAC, XCOFM);
                    s += ecl_range(phin, g.in1[i - n0], g.out1[i - n0])
                             ? 0.0 : v;
                }

                for (size_t i = c0; i < hi; i++) {
                    bool ecl = false;
                    for (int j = fg.moff[i - n1]; j < fg.moff[i - n1 + 1]; j++)
                        if (ecl_range(phin, fg.min_[j], fg.mout_[j])) {
                            ecl = true;
                            break;
                        }
                    if (!ecl)
                        s += s1_elem<LTYPE, BEAM>(g, i, ex, ey, ez, L, beam,
                                                  spin, VFAC, XCOFM);
                }
                acc[k] += s;
            }
        }
        #pragma omp critical
        for (int k = 0; k < nent; k++) out[k] += acc[k];
    }
}

template<int LTYPE, bool BEAM, bool GLENS>
void sum_star2_multi_t(const FlatGrid &fg, const Lcurve::PhaseBatch &pb,
                       const LdcC &L, double beam, double spin,
                       double VFAC, double XCOFM, double rlens1, double *out) {
    const GridPtrs g(fg);
    const size_t n0 = fg.n0, n1 = fg.n1, n = fg.n;
    const int nent = static_cast<int>(pb.size());
    const long nblk = static_cast<long>((n + FACE_BLOCK - 1) / FACE_BLOCK);
    std::fill(out, out + nent, 0.0);

    #pragma omp parallel if(!omp_in_parallel())
    {
        std::vector<double> acc(nent, 0.0);
        #pragma omp for schedule(dynamic) nowait
        for (long b = 0; b < nblk; b++) {
            const size_t lo = b * FACE_BLOCK;
            const size_t hi = std::min(n, lo + FACE_BLOCK);
            const size_t a1 = std::min(hi, n0);
            const size_t b0 = std::max(lo, n0);
            const size_t b1 = std::min(hi, n1);
            const size_t c0 = std::max(lo, n1);

            for (int k = 0; k < nent; k++) {
                const double ex = pb.ex[k], ey = pb.ey[k], ez = pb.ez[k];
                const double phin = pb.phin[k];
                double s = 0.0;

                #pragma omp simd reduction(+:s)
                for (size_t i = lo; i < a1; i++)
                    s += s2_elem<LTYPE, BEAM, GLENS>(g, i, ex, ey, ez, L,
                                                     beam, spin, VFAC,
                                                     XCOFM, rlens1);

                #pragma omp simd reduction(+:s)
                for (size_t i = b0; i < b1; i++) {
                    double v = s2_elem<LTYPE, BEAM, GLENS>(g, i, ex, ey, ez,
                                                           L, beam, spin,
                                                           VFAC, XCOFM,
                                                           rlens1);
                    s += ecl_range(phin, g.in1[i - n0], g.out1[i - n0])
                             ? 0.0 : v;
                }

                for (size_t i = c0; i < hi; i++) {
                    bool ecl = false;
                    for (int j = fg.moff[i - n1]; j < fg.moff[i - n1 + 1]; j++)
                        if (ecl_range(phin, fg.min_[j], fg.mout_[j])) {
                            ecl = true;
                            break;
                        }
                    if (!ecl)
                        s += s2_elem<LTYPE, BEAM, GLENS>(g, i, ex, ey, ez, L,
                                                         beam, spin, VFAC,
                                                         XCOFM, rlens1);
                }
                acc[k] += s;
            }
        }
        #pragma omp critical
        for (int k = 0; k < nent; k++) out[k] += acc[k];
    }
}

double flat_sum_star1(const FlatGrid &g, double ex, double ey, double ez,
                      double phin, const LDC &ldc, double beam, double spin,
                      double VFAC, double XCOFM) {
    LdcC L{ldc.c1(), ldc.c2(), ldc.c3(), ldc.c4(),
           1.0 - (ldc.c1() + ldc.c2() + ldc.c3() + ldc.c4()),
           ldc.mucrit_val()};
    bool claret = (ldc.type() == LDC::CLARET);
    bool bm = (beam != 0.0);
    if (claret)
        return bm ? sum_star1_t<1, true>(g, ex, ey, ez, phin, L, beam, spin, VFAC, XCOFM)
                  : sum_star1_t<1, false>(g, ex, ey, ez, phin, L, beam, spin, VFAC, XCOFM);
    return bm ? sum_star1_t<0, true>(g, ex, ey, ez, phin, L, beam, spin, VFAC, XCOFM)
              : sum_star1_t<0, false>(g, ex, ey, ez, phin, L, beam, spin, VFAC, XCOFM);
}

double flat_sum_star2(const FlatGrid &g, double ex, double ey, double ez,
                      double phin, const LDC &ldc, double beam, double spin,
                      double VFAC, double XCOFM, bool glens1, double rlens1) {
    LdcC L{ldc.c1(), ldc.c2(), ldc.c3(), ldc.c4(),
           1.0 - (ldc.c1() + ldc.c2() + ldc.c3() + ldc.c4()),
           ldc.mucrit_val()};
    bool claret = (ldc.type() == LDC::CLARET);
    bool bm = (beam != 0.0);
    if (glens1) {
        if (claret)
            return bm ? sum_star2_t<1, true, true>(g, ex, ey, ez, phin, L, beam, spin, VFAC, XCOFM, rlens1)
                      : sum_star2_t<1, false, true>(g, ex, ey, ez, phin, L, beam, spin, VFAC, XCOFM, rlens1);
        return bm ? sum_star2_t<0, true, true>(g, ex, ey, ez, phin, L, beam, spin, VFAC, XCOFM, rlens1)
                  : sum_star2_t<0, false, true>(g, ex, ey, ez, phin, L, beam, spin, VFAC, XCOFM, rlens1);
    }
    if (claret)
        return bm ? sum_star2_t<1, true, false>(g, ex, ey, ez, phin, L, beam, spin, VFAC, XCOFM, rlens1)
                  : sum_star2_t<1, false, false>(g, ex, ey, ez, phin, L, beam, spin, VFAC, XCOFM, rlens1);
    return bm ? sum_star2_t<0, true, false>(g, ex, ey, ez, phin, L, beam, spin, VFAC, XCOFM, rlens1)
              : sum_star2_t<0, false, false>(g, ex, ey, ez, phin, L, beam, spin, VFAC, XCOFM, rlens1);
}

} // unnamed namespace

void Lcurve::flat_sum_star1_multi(const FlatGrid &g, const PhaseBatch &pb,
                                  const LDC &ldc, double beam, double spin,
                                  double VFAC, double XCOFM, double *out) {
    LdcC L{ldc.c1(), ldc.c2(), ldc.c3(), ldc.c4(),
           1.0 - (ldc.c1() + ldc.c2() + ldc.c3() + ldc.c4()),
           ldc.mucrit_val()};
    bool claret = (ldc.type() == LDC::CLARET);
    bool bm = (beam != 0.0);
    if (claret) {
        if (bm) sum_star1_multi_t<1, true>(g, pb, L, beam, spin, VFAC, XCOFM, out);
        else    sum_star1_multi_t<1, false>(g, pb, L, beam, spin, VFAC, XCOFM, out);
    } else {
        if (bm) sum_star1_multi_t<0, true>(g, pb, L, beam, spin, VFAC, XCOFM, out);
        else    sum_star1_multi_t<0, false>(g, pb, L, beam, spin, VFAC, XCOFM, out);
    }
}

void Lcurve::flat_sum_star2_multi(const FlatGrid &g, const PhaseBatch &pb,
                                  const LDC &ldc, double beam, double spin,
                                  double VFAC, double XCOFM,
                                  bool glens1, double rlens1, double *out) {
    LdcC L{ldc.c1(), ldc.c2(), ldc.c3(), ldc.c4(),
           1.0 - (ldc.c1() + ldc.c2() + ldc.c3() + ldc.c4()),
           ldc.mucrit_val()};
    bool claret = (ldc.type() == LDC::CLARET);
    bool bm = (beam != 0.0);
    if (glens1) {
        if (claret) {
            if (bm) sum_star2_multi_t<1, true, true>(g, pb, L, beam, spin, VFAC, XCOFM, rlens1, out);
            else    sum_star2_multi_t<1, false, true>(g, pb, L, beam, spin, VFAC, XCOFM, rlens1, out);
        } else {
            if (bm) sum_star2_multi_t<0, true, true>(g, pb, L, beam, spin, VFAC, XCOFM, rlens1, out);
            else    sum_star2_multi_t<0, false, true>(g, pb, L, beam, spin, VFAC, XCOFM, rlens1, out);
        }
    } else {
        if (claret) {
            if (bm) sum_star2_multi_t<1, true, false>(g, pb, L, beam, spin, VFAC, XCOFM, rlens1, out);
            else    sum_star2_multi_t<1, false, false>(g, pb, L, beam, spin, VFAC, XCOFM, rlens1, out);
        } else {
            if (bm) sum_star2_multi_t<0, true, false>(g, pb, L, beam, spin, VFAC, XCOFM, rlens1, out);
            else    sum_star2_multi_t<0, false, false>(g, pb, L, beam, spin, VFAC, XCOFM, rlens1, out);
        }
    }
}

namespace {

/* ---- disc / edge / spot elements (same as comp_light.cpp) ---- */

inline double disc_elem(const Point &pt, const Subs::Vec3 &earth, double phi,
                        double lin_ld, double quad_ld) {
    if (!pt.visible(phi)) return 0.0;
    double mu = Subs::dot(earth, pt.dirn);
    if (mu <= 0.0) return 0.0;
    double om = 1.0 - mu;
    return mu * pt.flux * (1.0 - om * (lin_ld + quad_ld * om));
}

inline double spot_elem(const Point &pt, const Subs::Vec3 &earth, double phi) {
    if (!pt.visible(phi)) return 0.0;
    double mu = Subs::dot(earth, pt.dirn);
    return (mu > 0.0) ? mu * pt.flux : 0.0;
}

} // unnamed namespace

/* ---------------- public entry points ------------------------- */

double Lcurve::comp_light_flat(double iangle, const LDC &ldc1, const LDC &ldc2,
                               double lin_ld_disc, double quad_ld_disc,
                               double phase, double expose, int ndiv, double q,
                               double beam1, double beam2,
                               double spin1, double spin2, float vscale,
                               bool glens1, double rlens1, const Ginterp &gint,
                               const FlatGrid &star1f, const FlatGrid &star2f,
                               const FlatGrid &star1c, const FlatGrid &star2c,
                               const vector<Point> &disc,
                               const vector<Point> &edge,
                               const vector<Point> &spot) {
    const double XCOFM = q / (1.0 + q);
    const double VFAC = vscale / (Constants::C / 1.e3);
    const double ri = Subs::deg2rad(iangle);
    const double cosi = cos(ri), sini = sin(ri);

    double sum = 0.0;

    for (int nd = 0; nd < ndiv; ++nd) {
        double phi, wgt;
        if (ndiv == 1) { phi = phase; wgt = 1.0; }
        else {
            phi = phase + expose * (nd - (ndiv - 1) / 2.0) / (ndiv - 1);
            wgt = (nd == 0 || nd == ndiv - 1) ? 0.5 : 1.0;
        }

        Subs::Vec3 earth = Roche::set_earth(cosi, sini, phi);
        double ex = earth.x(), ey = earth.y(), ez = earth.z();
        double phin = phi - std::floor(phi);

        const FlatGrid &star1 = (gint.type(phi) == 1) ? star1f : star1c;
        const FlatGrid &star2 = (gint.type(phi) == 3) ? star2f : star2c;

        double s1 = flat_sum_star1(star1, ex, ey, ez, phin, ldc1, beam1,
                                   spin1, VFAC, XCOFM);
        double s2 = flat_sum_star2(star2, ex, ey, ez, phin, ldc2, beam2,
                                   spin2, VFAC, XCOFM, glens1, rlens1);

        double sd = 0., se = 0., ss = 0.;
        for (const auto &pt : disc)
            sd += disc_elem(pt, earth, phi, lin_ld_disc, quad_ld_disc);
        for (const auto &pt : edge)
            se += disc_elem(pt, earth, phi, lin_ld_disc, quad_ld_disc);
        for (const auto &pt : spot)
            ss += spot_elem(pt, earth, phi);

        sum += wgt * (gint.scale1(phi) * s1 + gint.scale2(phi) * s2 +
                      sd + se + ss);
    }

    return sum / std::max(1, ndiv - 1);
}

double Lcurve::comp_star1_flat(double iangle, const LDC &ldc1, double phase,
                               double expose, int ndiv, double q, double beam1,
                               float vscale, const Ginterp &gint,
                               const FlatGrid &star1f, const FlatGrid &star1c) {
    const double XCOFM = q / (1.0 + q);
    const double VFAC = vscale / (Constants::C / 1.e3);
    const double ri = Subs::deg2rad(iangle);
    const double cosi = cos(ri), sini = sin(ri);

    double sum = 0.0;
    for (int nd = 0; nd < ndiv; ++nd) {
        double phi, wgt;
        if (ndiv == 1) { phi = phase; wgt = 1.0; }
        else {
            phi = phase + expose * (nd - (ndiv - 1) / 2.0) / (ndiv - 1);
            wgt = (nd == 0 || nd == ndiv - 1) ? 0.5 : 1.0;
        }

        Subs::Vec3 earth = Roche::set_earth(cosi, sini, phi);
        double phin = phi - std::floor(phi);
        const FlatGrid &star1 = (gint.type(phi) == 1) ? star1f : star1c;

        // NB comp_star1 uses spin = 1 for the beaming term
        sum += wgt * gint.scale1(phi) *
               flat_sum_star1(star1, earth.x(), earth.y(), earth.z(), phin,
                              ldc1, beam1, 1.0, VFAC, XCOFM);
    }
    return sum / std::max(1, ndiv - 1);
}

double Lcurve::comp_star2_flat(double iangle, const LDC &ldc2, double phase,
                               double expose, int ndiv, double q, double beam2,
                               float vscale, bool glens1, double rlens1,
                               const Ginterp &gint,
                               const FlatGrid &star2f, const FlatGrid &star2c) {
    const double XCOFM = q / (1.0 + q);
    const double VFAC = vscale / (Constants::C / 1.e3);
    const double ri = Subs::deg2rad(iangle);
    const double cosi = cos(ri), sini = sin(ri);

    double sum = 0.0;
    for (int nd = 0; nd < ndiv; ++nd) {
        double phi, wgt;
        if (ndiv == 1) { phi = phase; wgt = 1.0; }
        else {
            phi = phase + expose * (nd - (ndiv - 1) / 2.0) / (ndiv - 1);
            wgt = (nd == 0 || nd == ndiv - 1) ? 0.5 : 1.0;
        }

        Subs::Vec3 earth = Roche::set_earth(cosi, sini, phi);
        double phin = phi - std::floor(phi);
        const FlatGrid &star2 = (gint.type(phi) == 3) ? star2f : star2c;

        // NB comp_star2 uses spin = 1 for the beaming term
        sum += wgt * gint.scale2(phi) *
               flat_sum_star2(star2, earth.x(), earth.y(), earth.z(), phin,
                              ldc2, beam2, 1.0, VFAC, XCOFM, glens1, rlens1);
    }
    return sum / std::max(1, ndiv - 1);
}
