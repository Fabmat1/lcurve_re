/**********************************************************************
 *  svdfit.cpp  –  consolidated implementation that works with
 *                 std::vector<std::vector<…>> instead of Buffer2D.
 *
 *  The three public overloads are identical to the originals from
 *  the caller’s point of view, but the common logic now lives in the
 *  template helper svdfit_core<>() further below.
 *
 *********************************************************************/

#include <vector>
#include <algorithm>   // std::max_element
#include <string>
#include "new_subs.h"  // defines rv, ddat, Subs_Error, svdcmp, svbksb, sqr, …

namespace Subs {

/*--------------------------------------------------------------------
 *  Small internal helper – all the hard work is done only once here.
 *------------------------------------------------------------------*/
template <class Real, class DataCont, class RowBuilder, class ModelFunc>
static Real svdfit_core(const DataCont&                  data,
                        std::vector<Real>&               a,
                        RowBuilder&&                     build_row,
                        ModelFunc&&                      model_value,
                        std::vector<std::vector<Real>>&  u,
                        std::vector<std::vector<Real>>&  v,
                        std::vector<Real>&               w)
{
    const std::size_t ndata = data.size();
    const std::size_t nc    = a.size();
    const Real        TOL   = Real(1e-5);

    /* -- number of valid (σ  > 0) points -- */
    std::size_t ndat = 0;
    for (const auto& d : data) if (d.z > 0) ++ndat;

    /* -- work arrays ------------------------------------------------ */
    u.assign(ndat, std::vector<Real>(nc));   // ndat × nc
    v.assign(nc,   std::vector<Real>(nc));   // nc   × nc
    w.resize(nc);

    std::vector<Real> b(ndat);

    /* -- build weighted design matrix U and RHS b ------------------- */
    std::size_t k = 0;
    for (std::size_t i = 0; i < ndata; ++i) {
        if (data[i].z > 0) {
            const Real invsig = Real(1) / data[i].z;
            build_row(i, k, invsig, u);          // fill U(k,·)
            b[k++] = invsig * data[i].y;         // RHS
        }
    }

    /* -- singular-value decomposition ------------------------------- */
    svdcmp(u, w, v);

    /* -- edit tiny singular values ---------------------------------- */
    const Real wmax   = *std::max_element(w.begin(), w.end());
    const Real thresh = TOL * wmax;
    for (auto& wi : w) if (wi < thresh) wi = Real(0);

    /* -- solve and obtain coefficients ------------------------------ */
    svbksb(u, w, v, b, a);

    /* -- χ² ---------------------------------------------------------- */
    Real chisq = Real(0);
    k = 0;
    for (std::size_t i = 0; i < ndata; ++i) {
        if (data[i].z > 0) {
            const Real model = model_value(i, a);
            chisq += sqr( (data[i].y - model) / data[i].z );
        }
    }
    return chisq;
}

/*--------------------------------------------------------------------
 *  1) float version, generic design matrix supplied in ‘vect’
 *------------------------------------------------------------------*/
double svdfit(const std::vector<rv>&                   data,
              std::vector<float>&                      a,
              const std::vector<std::vector<float>>&   vect,
              std::vector<std::vector<float>>&         u,
              std::vector<std::vector<float>>&         v,
              std::vector<float>&                      w)
{
    if (vect.empty())
        throw Subs_Error("svdfit[float]: function array is empty.");

    if (a.size() != vect[0].size())
        throw Subs_Error("svdfit[float]: number of coefficients = " +
                         std::to_string(a.size()) +
                         " does not match number of columns in function array = " +
                         std::to_string(vect[0].size()));

    if (data.size() != vect.size())
        throw Subs_Error("svdfit[float]: number of data points = " +
                         std::to_string(data.size()) +
                         " does not match number of rows in function array = " +
                         std::to_string(vect.size()));

    /* ---- callable that fills one row of U ------------------------- */
    auto build_row = [&](std::size_t i, std::size_t k, float invsig,
                         std::vector<std::vector<float>>& U)
    {
        for (std::size_t j = 0; j < a.size(); ++j)
            U[k][j] = invsig * vect[i][j];
    };

    /* ---- callable that returns the model value for χ² ------------- */
    auto model_value = [&](std::size_t i, const std::vector<float>& aa)->float
    {
        float sum = 0.f;
        for (std::size_t j = 0; j < aa.size(); ++j)
            sum += aa[j] * vect[i][j];
        return sum;
    };

    return svdfit_core<float>(data, a, build_row, model_value, u, v, w);
}

/*--------------------------------------------------------------------
 *  2) double version, generic design matrix supplied in ‘vect’
 *------------------------------------------------------------------*/
double svdfit(const std::vector<ddat>&                  data,
              std::vector<double>&                      a,
              const std::vector<std::vector<double>>&   vect,
              std::vector<std::vector<double>>&         u,
              std::vector<std::vector<double>>&         v,
              std::vector<double>&                      w)
{
    if (vect.empty())
        throw Subs_Error("svdfit[double]: function array is empty.");

    if (a.size() != vect[0].size())
        throw Subs_Error("svdfit[double]: number of coefficients = " +
                         std::to_string(a.size()) +
                         " does not match number of columns in function array = " +
                         std::to_string(vect[0].size()));

    if (data.size() != vect.size())
        throw Subs_Error("svdfit[double]: number of data points = " +
                         std::to_string(data.size()) +
                         " does not match number of rows in function array = " +
                         std::to_string(vect.size()));

    auto build_row = [&](std::size_t i, std::size_t k, double invsig,
                         std::vector<std::vector<double>>& U)
    {
        for (std::size_t j = 0; j < a.size(); ++j)
            U[k][j] = invsig * vect[i][j];
    };

    auto model_value = [&](std::size_t i, const std::vector<double>& aa)->double
    {
        double sum = 0.0;
        for (std::size_t j = 0; j < aa.size(); ++j)
            sum += aa[j] * vect[i][j];
        return sum;
    };

    return svdfit_core<double>(data, a, build_row, model_value, u, v, w);
}

/*--------------------------------------------------------------------
 *  3) float version specialised for a 3-parameter sinusoid
 *------------------------------------------------------------------*/
double svdfit(const std::vector<rv>&            data,
              std::vector<float>&               a,
              const std::vector<double>&        cosine,
              const std::vector<double>&        sine,
              std::vector<std::vector<float>>&  u,
              std::vector<std::vector<float>>&  v,
              std::vector<float>&               w)
{
    const std::size_t ndata = data.size();
    if (a.size() != 3)
        throw Subs_Error("svdfit[sinusoid]: exactly 3 coefficients expected.");

    if (cosine.size() != ndata || sine.size() != ndata)
        throw Subs_Error("svdfit[sinusoid]: cosine/sine arrays must have the same length as data.");

    /* --- build one row of U --------------------------------------- */
    auto build_row = [&](std::size_t i, std::size_t k, float invsig,
                         std::vector<std::vector<float>>& U)
    {
        U[k][0] = invsig;
        U[k][1] = invsig * static_cast<float>(cosine[i]);
        U[k][2] = invsig * static_cast<float>(sine[i]);
    };

    /* --- corresponding model value -------------------------------- */
    auto model_value = [&](std::size_t i, const std::vector<float>& aa)->float
    {
        return aa[0] +
               aa[1] * static_cast<float>(cosine[i]) +
               aa[2] * static_cast<float>(sine[i]);
    };

    return svdfit_core<float>(data, a, build_row, model_value, u, v, w);
}

}   // namespace Subs