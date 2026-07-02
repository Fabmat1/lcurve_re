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
    double sumsa=0., sumvol=0.;

    // Star 1.
    #pragma omp parallel for schedule(static) reduction(+:sumsa,sumvol)
    for(long unsigned int i=0; i<star1.size(); i++){
        const Point& pt = star1[i];
        Subs::Vec3 vec = pt.posn-cofm1;
        double r = vec.length();
        double rcosa = Subs::dot(pt.dirn, vec);

        // sum solid angle and 3x volume of all elements
        sumsa += pt.area*rcosa/std::pow(r,3);
        sumvol += pt.area*rcosa;
    }
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
    double sumsa=0., sumvol=0.;

    // Star 2.
    #pragma omp parallel for schedule(static) reduction(+:sumsa,sumvol)
    for(long unsigned int i=0; i<star2.size(); i++){
        const Point& pt = star2[i];
        Subs::Vec3 vec = pt.posn-cofm2;
        double r = vec.length();
        double rcosa = Subs::dot(pt.dirn, vec);

        // sum solid angle and 3x volume of all elements
        sumsa += pt.area*rcosa/std::pow(r,3);
        sumvol += pt.area*rcosa;
    }
    return std::pow(sumvol/sumsa,1./3.);
}