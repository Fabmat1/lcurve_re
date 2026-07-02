// chain_stats.h
// ═══════════════════════════════════════════════════════════════════════
//  Posterior-chain summary statistics shared by the MCMC and LM solvers.
//
//  Reports each parameter (and each derived physical quantity) as
//      value  +err_up / −err_down
//  where err_up/err_down are the distances from the reported value to the
//  84.1th / 15.9th posterior percentiles.  Derived quantities are computed
//  per-sample so their intervals inherit all parameter correlations —
//  never by adding per-side errors in quadrature (Barlow 2003).
//
//  The results are written into the augmented output JSON as a single
//  solver-agnostic "fit_results" block that ASTRA parses.
// ═══════════════════════════════════════════════════════════════════════
#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "physical_prior.h"

namespace ChainStats {

// p ∈ [0,1]; linear interpolation between order statistics.
inline double percentile(std::vector<double> v, double p)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double idx = p * (v.size() - 1);
    const size_t lo  = static_cast<size_t>(std::floor(idx));
    const size_t hi  = static_cast<size_t>(std::ceil(idx));
    const double w   = idx - lo;
    return v[lo] * (1.0 - w) + v[hi] * w;
}

struct Summary {
    double value    = 0.0;   // reported point estimate
    double median   = 0.0;
    double err_up   = 0.0;   // value → 84.1th percentile (≥ 0)
    double err_down = 0.0;   // 15.9th percentile → value (≥ 0)
};

// Summarize one trace around a given point estimate.  When the estimate
// lies outside the 16–84% interval (can happen when an optimizer sits on
// a boundary the posterior piles up against), the one-sided distance is
// clamped at zero — a genuinely one-sided interval.
inline Summary summarize(const std::vector<double>& trace, double value)
{
    Summary s;
    s.value    = value;
    s.median   = percentile(trace, 0.5);
    s.err_up   = std::max(0.0, percentile(trace, 0.841) - value);
    s.err_down = std::max(0.0, value - percentile(trace, 0.159));
    return s;
}

inline nlohmann::json to_json(const Summary& s)
{
    return {
        {"value",      s.value},
        {"median",     s.median},
        {"sigma",      0.5 * (s.err_up + s.err_down)},
        {"sigma_up",   s.err_up},
        {"sigma_down", s.err_down},
    };
}

// ── Derived physical quantities, per-sample ───────────────────────────
//
//  samples[k] holds the k-th posterior draw of the *varied* parameters;
//  idx_* map into it (−1 = fixed at the given fallback value).
//
struct DerivedTraces {
    std::vector<std::string>         names;
    std::vector<std::vector<double>> traces;   // one trace per name
};

inline DerivedTraces derived_traces(
    const std::vector<std::vector<double>>& samples,
    int idx_i, int idx_q, int idx_vs, int idx_r1,
    int idx_r2, int idx_t1, int idx_t2,
    double fix_i, double fix_q, double fix_vs, double fix_r1,
    double fix_r2, double fix_t1, double fix_t2,
    double P_days)
{
    using namespace PhysicalPrior;
    static const std::pair<const char*, int> quantities[] = {
        {"K1_km_s",      DRV_K1},
        {"K2_km_s",      DRV_K2},
        {"R1_Rsun",      DRV_R1},
        {"R2_Rsun",      DRV_R2},
        {"M1_Msun",      DRV_M1},
        {"M2_Msun",      DRV_M2},
        {"M_total_Msun", DRV_MTOTAL},
        {"logg1_dex",    DRV_LOGG1},
        {"logg2_dex",    DRV_LOGG2},
        {"a_Rsun",       DRV_A_RSUN},
    };

    DerivedTraces out;
    for (auto& [name, tag] : quantities) {
        out.names.emplace_back(name);
        out.traces.emplace_back();
        out.traces.back().reserve(samples.size());
    }

    auto pick = [](const std::vector<double>& s, int idx, double fix) {
        return idx >= 0 ? s[idx] : fix;
    };

    for (const auto& s : samples) {
        const double i  = pick(s, idx_i,  fix_i);
        const double q  = pick(s, idx_q,  fix_q);
        const double vs = pick(s, idx_vs, fix_vs);
        const double r1 = pick(s, idx_r1, fix_r1);
        const double r2 = pick(s, idx_r2, fix_r2);
        const double t1 = pick(s, idx_t1, fix_t1);
        const double t2 = pick(s, idx_t2, fix_t2);
        for (size_t k = 0; k < std::size(quantities); ++k)
            out.traces[k].push_back(PhysicalPrior::eval_derived_tag(
                quantities[k].second, i, q, vs, r1, r2, t1, t2, P_days));
    }
    return out;
}

// ── Full fit_results block ────────────────────────────────────────────
//
//  point[j]: the reported value for parameter j (LM optimum or posterior
//  median).  samples: post-burn-in draws, samples[k][j].
//
inline nlohmann::json build_fit_results(
    const std::string& method,
    const std::vector<std::string>& names,
    const std::vector<double>& point,
    const std::vector<std::vector<double>>& samples,
    double chisq_lc, int ndata, int npar,
    double chi2_scale)
{
    nlohmann::json out;
    out["method"]     = method;
    out["n_samples"]  = samples.size();
    out["chi2_scale"] = chi2_scale;
    out["best_chisq_lc"] = chisq_lc;
    out["reduced_chi2"]  = chisq_lc / std::max(1, ndata - npar);

    std::vector<double> trace(samples.size());
    for (size_t j = 0; j < names.size(); ++j) {
        for (size_t k = 0; k < samples.size(); ++k)
            trace[k] = samples[k][j];
        const Summary s = summarize(trace, point[j]);
        out["best_pars"][names[j]]  = s.value;
        out["median"][names[j]]     = s.median;
        out["sigma"][names[j]]      = 0.5 * (s.err_up + s.err_down);
        out["sigma_up"][names[j]]   = s.err_up;
        out["sigma_down"][names[j]] = s.err_down;
    }
    return out;
}

// Append per-sample derived quantities (point estimate re-evaluated at the
// reported parameter values so value and interval share one convention).
inline void add_implied(nlohmann::json& fit_results,
                        const DerivedTraces& dt,
                        const std::vector<double>& point_values)
{
    for (size_t k = 0; k < dt.names.size(); ++k) {
        const Summary s = summarize(dt.traces[k], point_values[k]);
        fit_results["implied"][dt.names[k]] = to_json(s);
    }
}

} // namespace ChainStats
