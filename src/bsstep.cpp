//  bsstep.cpp   – modern, compact replacement
//  --------------------------------------------------------------
//  still depends on the Numerical-Recipes helpers  mmid, stoerm
//  and   pzextr  plus  Subs::sqr()  that you already have.
//  The global work-space expected by pzextr is kept.
//
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include "new_subs.h"

// -----------------------------------------------------------------
//  Global work space, kept for pzextr compatibility
// -----------------------------------------------------------------
double **d = nullptr;   // rows point into a contiguous pool we allocate
double  *x = nullptr;   // Bulirsch–Stoer x-values needed by pzextr


struct BsState {
    int    first  = 1;      
    int    kmax   = 0;
    int    kopt   = 0;
    double epsold = -1.0;
    double xnew   = 0.0;
};

// =================================================================
//  One single implementation that does the real work --------------
// =================================================================
template<class Deriv, class Midpoint>
static bool
bsstep_impl(              //  returns  true  if h under-flowed   (same as NR code)
        double           y[],     double dydx[], int nv,
        double          &xx,      double  htry,  double  eps,
        double           yscal[],
        double          &hdid,    double &hnext,
        const Deriv&     derivs,
        const Midpoint&  mid,                 // mmid  or stoerm
        const std::vector<int>& nseq,         // {2,4,6,...} or {1,2,3,...}
        BsState&         st)
{
    constexpr double SAFE1  = 0.25;
    constexpr double SAFE2  = 0.7;
    constexpr double REDMAX = 1.0e-5;
    constexpr double REDMIN = 0.7;
    constexpr double TINY   = 1.0e-30;
    constexpr double SCALMX = 0.1;

    const int KMAXX = static_cast<int>(nseq.size());
    const int IMAXX = KMAXX + 1;

    //-----------------------------------------------------------------
    // Local work arrays  (vectors so no VLAs / no manual delete)
    //-----------------------------------------------------------------
    std::vector<double>          yerr (nv),
                                 ysav (nv),
                                 yseq (nv),
                                 err  (KMAXX);

    std::vector<double>          a    (IMAXX);
    std::vector<std::vector<double>>
                                 alf  (KMAXX, std::vector<double>(KMAXX));

    //-----------------------------------------------------------------
    // Global work space expected by pzextr
    //-----------------------------------------------------------------
    std::vector<double>          dpool(nv * KMAXX);
    std::vector<double*>         drows(nv);
    for (int i = 0; i < nv; ++i) drows[i] = dpool.data() + i * KMAXX;

    std::vector<double>          xvec (KMAXX);

    ::d = drows.data();
    ::x = xvec.data();

    //-----------------------------------------------------------------
    // Initialisation that depends on eps – exactly the Numerical-Recipes
    //-----------------------------------------------------------------
    if (eps != st.epsold) {
        hnext     = st.xnew = -1.0e29;
        const double eps1   = SAFE1 * eps;

        a[0] = nseq[0] + 1;
        for (int k = 0; k < KMAXX; ++k) a[k + 1] = a[k] + nseq[k + 1];

        for (int iq = 1; iq < KMAXX; ++iq)
            for (int k = 0; k < iq; ++k)
                alf[k][iq] = std::pow(eps1,
                                      (a[k + 1] - a[iq + 1]) /
                                     ((a[iq + 1] - a[0] + 1.0) * (2 * k + 3)));

        st.epsold = eps;
        for (st.kopt = 1; st.kopt < KMAXX - 1; ++st.kopt)
            if (a[st.kopt + 1] > a[st.kopt] * alf[st.kopt - 1][st.kopt])
                break;
        st.kmax = st.kopt;
    }

    //-----------------------------------------------------------------
    // Copy input, decide whether we have to restart the step
    //-----------------------------------------------------------------
    double h        = htry;
    std::copy(y, y + nv, ysav.begin());
    if (xx != st.xnew || h != hnext) { st.first = 1; st.kopt = st.kmax; }

    int    reduct    = 0;
    int    km        = 0;
    double red       = 0.0;
    bool   exitflag  = false;

    //-----------------------------------------------------------------
    //  Main attempt loop – identical to Numerical-Recipes, only shorter
    //-----------------------------------------------------------------
    for (;;) {
        for (int k = 0; k <= st.kmax; ++k) {
            st.xnew = xx + h;
            if (st.xnew == xx) return true;                   // step underflow

            mid(ysav.data(), dydx, nv, xx, h, nseq[k],
                yseq.data(), derivs);                         // mmid / stoerm

            const double xest = Subs::sqr(h / nseq[k]);
            Subs::pzextr(k, xest, yseq.data(), y, yerr.data(), nv);

            if (k != 0) {
                double errmax = TINY;
                for (int i = 0; i < nv; ++i)
                    errmax = std::max(errmax,
                                      std::fabs(yerr[i] / yscal[i]));
                errmax /= eps;
                km       = k - 1;
                err[km]  = std::pow(errmax / SAFE1, 1.0 / (2 * km + 3));

                if (k >= st.kopt - 1 || st.first) {
                    if (errmax < 1.0) { exitflag = true; break; }
                    if (k == st.kmax || k == st.kopt + 1)      red = SAFE2 / err[km];
                    else if (k == st.kopt && alf[st.kopt - 1][st.kopt] < err[km])
                                                              red = 1.0 / err[km];
                    else if (st.kopt == st.kmax &&
                             alf[km][st.kmax - 1] < err[km])  red = alf[km][st.kmax - 1] *
                                                                    SAFE2 / err[km];
                    else if (alf[km][st.kopt] < err[km])       red = alf[km][st.kopt - 1] /
                                                                    err[km];
                    if (red) break;
                }
            }
        }
        if (exitflag) break;

        red      = std::clamp(red, REDMAX, REDMIN);
        h       *= red;
        reduct   = 1;
    }

    xx      = st.xnew;
    hdid    = h;
    st.first = 0;

    double wrkmin = 1.0e35, scale = 0.0, fact;

    for (int kk = 0; kk <= km; ++kk) {
        fact  = std::max(err[kk], SCALMX);
        double work = fact * a[kk + 1];
        if (work < wrkmin) { wrkmin = work; scale = fact; st.kopt = kk + 1; }
    }
    hnext = h / scale;

    if (st.kopt >= km && st.kopt != st.kmax && !reduct) {
        fact = std::max(scale / alf[st.kopt - 1][st.kopt], SCALMX);
        if (a[st.kopt + 1] * fact <= wrkmin) {
            hnext = h / fact;
            ++st.kopt;
        }
    }
    return false;
}

// =================================================================
//  Three thin wrappers – exactly the public interface you had
// =================================================================
bool Subs::bsstep(double y[], double dydx[], int nv,
                  double& xx, double htry, double eps,
                  double yscal[],
                  double& hdid, double& hnext,
                  void (*derivs)(double, double[], double[]))
{
    static BsState state;                          // keeps its values
    static const std::vector<int> seq{2,4,6,8,10,12,14,16};

    auto mid = [](double y[], double dydx[], int nv, double x, double h,
                  int nstep, double yout[], void (*f)(double,double[],double[]))
               {
                   mmid(y, dydx, nv, x, h, nstep, yout, f);
               };
    return bsstep_impl(y, dydx, nv, xx, htry, eps, yscal,
                       hdid, hnext, derivs, mid, seq, state);
}

bool Subs::bsstep(double y[], double dydx[], int nv,
                  double& xx, double htry, double eps,
                  double yscal[],
                  double& hdid, double& hnext,
                  const Bsfunc& derivs)
{
    static BsState state;
    static const std::vector<int> seq{2,4,6,8,10,12,14,16};

    auto mid = [](double y[], double dydx[], int nv, double x, double h,
                  int nstep, double yout[], const Bsfunc& f)
               {
                   mmid(y, dydx, nv, x, h, nstep, yout, f);
               };
    return bsstep_impl(y, dydx, nv, xx, htry, eps, yscal,
                       hdid, hnext, derivs, mid, seq, state);
}

bool Subs::bsstepst(double y[], double dydx[], int nv,
                    double& xx, double htry, double eps,
                    double yscal[],
                    double& hdid, double& hnext,
                    const Bsfunc& derivs)
{
    static BsState state;
    static const std::vector<int> seq{1,2,3,4,5,6,7,8,9,10,11,12};

    auto mid = [](double y[], double dydx[], int nv, double x, double h,
                  int nstep, double yout[], const Bsfunc& f)
               {
                   stoerm(y, dydx, nv, x, h, nstep, yout, f);
               };
    return bsstep_impl(y, dydx, nv, xx, htry, eps, yscal,
                       hdid, hnext, derivs, mid, seq, state);
}