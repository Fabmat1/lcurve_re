#ifndef LCURVE_COMP_LIGHT_CUDA_H
#define LCURVE_COMP_LIGHT_CUDA_H

#include <cstdint>

#include "lcurve.h"

namespace Lcurve {

// CUDA implementations of the batched stellar-flux sweeps. They return
// false when CUDA is disabled, unavailable, or the batch is too small to
// amortise a launch; callers then run the existing CPU implementation.
bool cuda_sum_star1_multi(const FlatGrid &grid, const PhaseBatch &phases,
                          const LDC &ldc, double beam, double spin,
                          double vfac, double xcofm, double *out);

bool cuda_sum_star2_multi(const FlatGrid &grid, const PhaseBatch &phases,
                          const LDC &ldc, double beam, double spin,
                          double vfac, double xcofm, bool glens1,
                          double rlens1, double *out);

// Successful batched stellar-flux sweeps offloaded by this thread.
std::uint64_t cuda_flux_evaluation_count();

} // namespace Lcurve

#endif
