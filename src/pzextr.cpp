#include <vector>      // std::vector
#include <algorithm>   // std::copy
#include "new_subs.h"  // keeps the original public interface

//  These two symbols still come from bsstep exactly as before.
extern double **d;   // d[j][k] : Neville tableau
extern double  *x;   // x[k]    : previous xest-values

/*
 * Subs::pzextr  –  polynomial extrapolation used by Bulirsch–Stoer.
 *
 * Parameters are identical to the legacy routine:
 *   iest : current column (0 … n)
 *   xest : step size that produced yest
 *   yest : raw solution produced by the stepper     (input)
 *   yz   : extrapolated solution                    (output)
 *   dy   : error estimate on yz                     (output)
 *   nv   : number of ODE components
 *
 * All arithmetic and memory complexity is unchanged.
 */
void Subs::pzextr(int  iest,
                  double xest,
                  double yest[],
                  double yz[],
                  double dy[],
                  int    nv)
{
    // --- 1. Temporary work array (replaces illegal VLA) ---------------
    std::vector<double> c;          // constructed only when needed
    if (iest != 0) c.assign(yest, yest + nv);

    // --- 2. Initial copy: dy = yz = yest ------------------------------
    std::copy(yest, yest + nv, dy);
    std::copy(yest, yest + nv, yz);

    // --- 3. Store current x ------------------------------------------
    x[iest] = xest;

    // --- 4. First column is trivial ----------------------------------
    if (iest == 0) {
        for (int j = 0; j < nv; ++j)
            d[j][0] = yest[j];
        return;
    }

    // --- 5. Neville–Aitken tableau -----------------------------------
    for (int k1 = 0; k1 < iest; ++k1) {
        const double delta_inv = 1.0 / (x[iest - k1 - 1] - xest);
        const double f1        = xest * delta_inv;
        const double f2        = x[iest - k1 - 1] * delta_inv;

        for (int j = 0; j < nv; ++j) {
            const double q = d[j][k1];
            d[j][k1]       = dy[j];

            const double delta = c[j] - q;
            dy[j]  = f1 * delta;
            c[j]   = f2 * delta;
            yz[j] += dy[j];
        }
    }

    // --- 6. Store last column ----------------------------------------
    for (int j = 0; j < nv; ++j)
        d[j][iest] = dy[j];
}