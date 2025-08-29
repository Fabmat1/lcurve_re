#include <cmath>
#include <vector>
#include "../new_subs.h"
#include "constants.h"
#include "../lroche_base/roche.h"
#include "lcurve.h"

#ifdef _OPENMP
#   include <omp.h>
#endif

using std::vector;

/* =============================================================
   Elementary helpers – no global state is modified anywhere
   ============================================================= */

namespace {

/* ---------- star-1 element ----------------------------------- */
template<class LDC_T>
inline double star1_elem(const Lcurve::Point& pt,
                         const Subs::Vec3&    earth,
                         double               phi,
                         const LDC_T&         ldc,
                         double               beam,
                         double               spin,
                         double               VFAC,
                         double               XCOFM)
{
    if (!pt.visible(phi)) return 0.0;

    double mu = Subs::dot(earth, pt.dirn);
    if (!ldc.see(mu))     return 0.0;

    if (beam == 0.0)                       /* no Doppler beaming         */
        return mu * pt.flux * ldc.imu(mu);

    /* Doppler beaming terms */
    double vx  = -VFAC * spin * pt.posn.y();
    double vy  =  VFAC * (spin*pt.posn.x() - XCOFM);
    double vr  = -(earth.x()*vx + earth.y()*vy);
    double vn  =  pt.dirn.x()*vx + pt.dirn.y()*vy;
    double mud =  mu - mu*vr - vn;

    return mu * pt.flux * (1.0 - beam*vr) * ldc.imu(mud);
}

/* ---------- star-2 element ----------------------------------- */
template<class LDC_T>
inline double star2_elem(const Lcurve::Point& pt,
                         const Subs::Vec3&    earth,
                         double               phi,
                         const LDC_T&         ldc,
                         double               beam,
                         double               spin,
                         double               VFAC,
                         double               XCOFM,
                         bool                 glens1,
                         double               rlens1)
{
    if (!pt.visible(phi)) return 0.0;

    double mu = Subs::dot(earth, pt.dirn);
    if (!ldc.see(mu))     return 0.0;

    /* gravitational lensing by star 1 (optional) */
    double magn = 1.0;
    if (glens1) {
        Subs::Vec3 s = pt.posn;
        double     d = -Subs::dot(s, earth);
        if (d > 0.0) {
            double p   = (s + d*earth).length();
            double ph  = 0.5*p;
            double rd  = rlens1*d;
            double pd  = (ph*ph > 25.0*rd) ? p + rd/p
                                           : ph + std::sqrt(ph*ph + rd);
            magn = pd*pd/(pd-ph)/ph/4.0;
        }
    }

    if (beam == 0.0)
        return mu * magn * pt.flux * ldc.imu(mu);

    double vx  = -VFAC * spin * pt.posn.y();
    double vy  =  VFAC * ( spin*(pt.posn.x()-1.0) + 1.0 - XCOFM );
    double vr  = -(earth.x()*vx + earth.y()*vy);
    double vn  =  pt.dirn.x()*vx + pt.dirn.y()*vy;
    double mud =  mu - mu*vr - vn;

    return mu * magn * pt.flux * (1.0 - beam*vr) * ldc.imu(mud);
}

/* ---------- disc / edge element ------------------------------ */
inline double disc_elem(const Lcurve::Point& pt,
                        const Subs::Vec3&    earth,
                        double               phi,
                        double               lin_ld,
                        double               quad_ld)
{
    if (!pt.visible(phi)) return 0.0;

    double mu = Subs::dot(earth, pt.dirn);
    if (mu <= 0.0) return 0.0;

    double om = 1.0 - mu;
    return mu * pt.flux * (1.0 - om*(lin_ld + quad_ld*om));
}

/* ---------- spot element ------------------------------------- */
inline double spot_elem(const Lcurve::Point& pt,
                        const Subs::Vec3&    earth,
                        double               phi)
{
    if (!pt.visible(phi)) return 0.0;

    double mu = Subs::dot(earth, pt.dirn);
    return (mu > 0.0) ? mu*pt.flux : 0.0;
}

/* Convenience: return deg->rad once */
inline void earth_vec(double iangle, double phi,
                      double &cosi,  double &sini,
                      Subs::Vec3 &earth)
{
    double ri = Subs::deg2rad(iangle);
    cosi = cos(ri);     sini = sin(ri);
    earth = Roche::set_earth(cosi, sini, phi);
}

}       // unnamed namespace



/* =============================================================
                comp_light  (all components)
   ============================================================= */

double Lcurve::comp_light(double iangle, const LDC& ldc1, const LDC& ldc2,
                          double lin_ld_disc, double quad_ld_disc,
                          double phase, double expose, int ndiv,
                          double q,
                          double beam1, double beam2,
                          double spin1, double spin2,
                          float  vscale,
                          bool   glens1, double rlens1,
                          const  Ginterp& gint,
                          const  vector<Point>& star1f,
                          const  vector<Point>& star2f,
                          const  vector<Point>& star1c,
                          const  vector<Point>& star2c,
                          const  vector<Point>& disc,
                          const  vector<Point>& edge,
                          const  vector<Point>& spot)
{
    const double XCOFM = q/(1.0+q);
    const double VFAC  = vscale/(Constants::C/1.e3);

    double sum = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum) schedule(static) \
                     if(!omp_in_parallel())
#endif
    for (int nd = 0; nd < ndiv; ++nd) {

        /* ---- sub-exposure phase & trapezoidal weight -------- */
        double phi, wgt;
        if (ndiv == 1) { phi = phase; wgt = 1.0; }
        else {
            phi = phase + expose*(nd - (ndiv-1)/2.0)/(ndiv-1);
            wgt = (nd==0 || nd==ndiv-1) ? 0.5 : 1.0;
        }

        /* ---- direction to the observer ---------------------- */
        double cosi, sini;
        Subs::Vec3 earth;
        earth_vec(iangle, phi, cosi, sini, earth);

        /* ---- choose fine / coarse grids --------------------- */
        const vector<Point>& star1 = (gint.type(phi)==1) ? star1f : star1c;
        const vector<Point>& star2 = (gint.type(phi)==3) ? star2f : star2c;

        /* ---- accumulate fluxes ------------------------------ */
        double s1=0., s2=0., sd=0., se=0., ss=0.;

        for (const auto& pt : star1)
            s1 += star1_elem(pt, earth, phi, ldc1,
                             beam1, spin1, VFAC, XCOFM);

        for (const auto& pt : star2)
            s2 += star2_elem(pt, earth, phi, ldc2,
                             beam2, spin2, VFAC, XCOFM,
                             glens1, rlens1);

        for (const auto& pt : disc)
            sd += disc_elem(pt, earth, phi, lin_ld_disc, quad_ld_disc);

        for (const auto& pt : edge)
            se += disc_elem(pt, earth, phi, lin_ld_disc, quad_ld_disc);

        for (const auto& pt : spot)
            ss += spot_elem(pt, earth, phi);

        sum += wgt * ( gint.scale1(phi)*s1 +
                       gint.scale2(phi)*s2 + sd + se + ss );
    }

    return sum / std::max(1, ndiv-1);
}



/* =============================================================
                  comp_star1  (white dwarf)
   ============================================================= */

double Lcurve::comp_star1(double iangle, const LDC& ldc1,
                          double phase, double expose, int ndiv,
                          double q, double beam1, float vscale,
                          const Ginterp& gint,
                          const vector<Point>& star1f,
                          const vector<Point>& star1c)
{
    const double XCOFM = q/(1.0+q);
    const double VFAC  = vscale/(Constants::C/1.e3);

    double sum = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum) schedule(static) \
                     if(!omp_in_parallel())
#endif
    for (int nd = 0; nd < ndiv; ++nd) {

        double phi, wgt;
        if (ndiv==1){ phi=phase; wgt=1.0; }
        else {
            phi = phase + expose*(nd - (ndiv-1)/2.0)/(ndiv-1);
            wgt = (nd==0||nd==ndiv-1)?0.5:1.0;
        }

        double cosi, sini;
        Subs::Vec3 earth;
        earth_vec(iangle, phi, cosi, sini, earth);

        const vector<Point>& star1 = (gint.type(phi)==1) ? star1f : star1c;

        double s1 = 0.0;
        for (const auto& pt : star1)
            s1 += star1_elem(pt, earth, phi, ldc1,
                             beam1, 1.0, VFAC, XCOFM);

        sum += wgt * gint.scale1(phi) * s1;
    }

    return sum / std::max(1, ndiv-1);
}



/* =============================================================
                     comp_star2  (secondary)
   ============================================================= */

double Lcurve::comp_star2(double iangle, const LDC& ldc2,
                          double phase, double expose, int ndiv,
                          double q, double beam2, float vscale,
                          bool   glens1, double rlens1,
                          const  Ginterp& gint,
                          const  vector<Point>& star2f,
                          const  vector<Point>& star2c)
{
    const double XCOFM = q/(1.0+q);
    const double VFAC  = vscale/(Constants::C/1.e3);

    double sum = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum) schedule(static) \
                     if(!omp_in_parallel())
#endif
    for (int nd = 0; nd < ndiv; ++nd) {

        double phi, wgt;
        if (ndiv==1){ phi=phase; wgt=1.0; }
        else {
            phi = phase + expose*(nd - (ndiv-1)/2.0)/(ndiv-1);
            wgt = (nd==0||nd==ndiv-1)?0.5:1.0;
        }

        double cosi, sini;
        Subs::Vec3 earth;
        earth_vec(iangle, phi, cosi, sini, earth);

        const vector<Point>& star2 = (gint.type(phi)==3) ? star2f : star2c;

        double s2 = 0.0;
        for (const auto& pt : star2)
            s2 += star2_elem(pt, earth, phi, ldc2,
                             beam2, 1.0, VFAC, XCOFM,
                             glens1, rlens1);

        sum += wgt * gint.scale2(phi) * s2;
    }

    return sum / std::max(1, ndiv-1);
}



/* =============================================================
        comp_disc  /  comp_edge  (just geometry differs)
   ============================================================= */

static double disc_like(double iangle,
                        double lin_ld, double quad_ld,
                        double phase, double expose, int ndiv,
                        const vector<Lcurve::Point>& surf)
{
    double sum = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum) schedule(static) \
                     if(!omp_in_parallel())
#endif
    for (int nd = 0; nd < ndiv; ++nd) {

        double phi, wgt;
        if (ndiv==1){ phi=phase; wgt=1.0; }
        else {
            phi = phase + expose*(nd - (ndiv-1)/2.0)/(ndiv-1);
            wgt = (nd==0||nd==ndiv-1)?0.5:1.0;
        }

        double cosi, sini;
        Subs::Vec3 earth;
        earth_vec(iangle, phi, cosi, sini, earth);

        double s = 0.0;
        for (const auto& pt : surf)
            s += disc_elem(pt, earth, phi, lin_ld, quad_ld);

        sum += wgt * s;
    }

    return sum / std::max(1, ndiv-1);
}

double Lcurve::comp_disc(double iangle,double lin,double quad,
                         double phase,double expose,int ndiv,double /*q*/,
                         const vector<Point>& disc)
{ return disc_like(iangle, lin, quad, phase, expose, ndiv, disc); }

double Lcurve::comp_edge(double iangle,double lin,double quad,
                         double phase,double expose,int ndiv,double /*q*/,
                         const vector<Point>& edge)
{ return disc_like(iangle, lin, quad, phase, expose, ndiv, edge); }



/* =============================================================
                       comp_spot
   ============================================================= */

double Lcurve::comp_spot(double iangle,
                         double phase,double expose,int ndiv,double /*q*/,
                         const vector<Point>& spot)
{
    double sum = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum) schedule(static) \
                     if(!omp_in_parallel())
#endif
    for (int nd=0; nd<ndiv; ++nd) {

        double phi, wgt;
        if (ndiv==1){ phi=phase; wgt=1.0; }
        else {
            phi = phase + expose*(nd - (ndiv-1)/2.0)/(ndiv-1);
            wgt = (nd==0||nd==ndiv-1)?0.5:1.0;
        }

        double cosi, sini;
        Subs::Vec3 earth;
        earth_vec(iangle, phi, cosi, sini, earth);

        double s=0.0;
        for (const auto& pt : spot)
            s += spot_elem(pt, earth, phi);

        sum += wgt*s;
    }

    return sum / std::max(1, ndiv-1);
}