#ifndef LCURVE_SET_STAR_GRID_CUDA_H
#define LCURVE_SET_STAR_GRID_CUDA_H

#include <cstdint>

#include "lcurve.h"

namespace Lcurve {

// The whole-grid CUDA path is opt-in through LCURVE_CUDA_GRID=1. Keeping the
// query separate avoids constructing temporary SoA inputs on the CPU fallback.
bool cuda_star_grid_enabled();

bool cuda_build_star_faces(
    std::vector<Point> &star, int first_face,
    const std::vector<double> &dx, const std::vector<double> &dy,
    const std::vector<double> &dz, const std::vector<double> &area_scale,
    Roche::STAR which_star, double q, double iangle, double r1, double r2,
    double rref1, double rref2, bool roche1, bool roche2,
    double spin1, double spin2, bool eclipse, double gref,
    double pref1, double pref2, double delta);

// Successful star-grid builds offloaded by this thread.
std::uint64_t cuda_grid_evaluation_count();

} // namespace Lcurve

#endif
