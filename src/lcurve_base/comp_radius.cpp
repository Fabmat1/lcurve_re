#include <cstdlib>
#include <iostream>
#include <vector>
#include "../new_subs.h"
#include "constants.h"
#include "../lroche_base/roche.h"
#include "lcurve.h"

/**
 * comp_radius1 computes a volume-averaged scaled radius for star 1.
 *
 * \param star1 -- grid for star1
 * \return the value of volume-averaged R1/a
 */

double Lcurve::comp_radius1(const vector<Lcurve::Point>& star1){

    const Subs::Vec3 cofm1(0.,0.,0.);

    // Star 1. Block partial sums merged in fixed order so the result does
    // not depend on thread count or scheduling.
    const size_t BLK = 8192, nn = star1.size(), nblk = (nn + BLK - 1)/BLK;
    std::vector<double> psa(nblk), pvol(nblk);
    #pragma omp parallel for schedule(static)
    for(long b=0; b<(long)nblk; b++){
        double bsa=0., bvol=0.;
        for(size_t i=b*BLK, e=std::min(nn, b*BLK+BLK); i<e; i++){
            const Point& pt = star1[i];
            Subs::Vec3 vec = pt.posn-cofm1;
            double r = vec.length();
            double rcosa = Subs::dot(pt.dirn, vec);

            // sum solid angle and 3x volume of all elements
            bsa += pt.area*rcosa/std::pow(r,3);
            bvol += pt.area*rcosa;
        }
        psa[b] = bsa; pvol[b] = bvol;
    }
    double sumsa=0., sumvol=0.;
    for(size_t b=0; b<nblk; b++){ sumsa += psa[b]; sumvol += pvol[b]; }
    return std::pow(sumvol/sumsa,1./3.);
}

/**
 * comp_radius2 computes a volume-averaged scaled radius for star 2.
 *
 * \param star2 -- grid for star2
 * \return the value of volume-averaged R2/a
 */

double Lcurve::comp_radius2(const vector<Lcurve::Point>& star2){

    const Subs::Vec3 cofm2(1.,0.,0.);

    // Star 2. Deterministic blocked merge (see comp_radius1).
    const size_t BLK = 8192, nn = star2.size(), nblk = (nn + BLK - 1)/BLK;
    std::vector<double> psa(nblk), pvol(nblk);
    #pragma omp parallel for schedule(static)
    for(long b=0; b<(long)nblk; b++){
        double bsa=0., bvol=0.;
        for(size_t i=b*BLK, e=std::min(nn, b*BLK+BLK); i<e; i++){
            const Point& pt = star2[i];
            Subs::Vec3 vec = pt.posn-cofm2;
            double r = vec.length();
            double rcosa = Subs::dot(pt.dirn, vec);

            // sum solid angle and 3x volume of all elements
            bsa += pt.area*rcosa/std::pow(r,3);
            bvol += pt.area*rcosa;
        }
        psa[b] = bsa; pvol[b] = bvol;
    }
    double sumsa=0., sumvol=0.;
    for(size_t b=0; b<nblk; b++){ sumsa += psa[b]; sumvol += pvol[b]; }
    return std::pow(sumvol/sumsa,1./3.);
}