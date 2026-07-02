#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>
#include "../new_subs.h"
#include "roche.h"

/**
 * 'face' computes the position and orientation of a face on either star in a binary assuming Roche geometry given
 * a direction, a reference radius and a potential.
 *
 * \param q    the mass ratio = M2/M1.
 * \param star specifies which star, primary or secondary is under consideration.
 * \param spin ratio of star in questions spin to the orbital frequency
 * \param dirn the direction (unit) vector from the centre of mass of the secondary to the face in question.
 * \param rref reference radius. This is a radius large enough to guarantee crossing of the reference potential. See ref_sphere
 * \param pref reference potential. This defines the precise location of the face.
 * \param acc  location accuracy (units of separation)
 * \param pvec position vector of centre of face (position vector in standard binary coordinates), returned
 * \param dvec orientation vector perpendicular to face, returned
 * \param r    distance from centre of mass of star, returned
 * \param g    magnitude of gravity at face, returned
 * \exception The routine throws exceptions if it cannot bracket the reference potential. This can occur if the reference radius fails to enclose
 * the face in question, or if the face is so deep in the potential that the initial search fails to reach it. Finally if acc is set too low an
 * exception may be thrown if too many binary chops occur. The behaviour at the L1 point is undefined so do not try to call it there.
 */

namespace {

// Roche potential along the ray cofm + r*dirn with the q- and spin-derived
// constants hoisted out of the root-finding loop. The expressions follow
// rpot1/rpot2 operation-for-operation so results are bit-identical.
struct RayPot {
    double mu, comp, spin_sq;
    double cx;      // x of the centre of mass (0 primary, 1 secondary)
    double dx, dy, dz;
    bool primary;

    RayPot(double q, Roche::STAR star, double spin, const Subs::Vec3 &dirn)
        : mu(q / (1 + q)), comp(1. - mu), spin_sq(spin * spin),
          cx(star == Roche::PRIMARY ? 0. : 1.),
          dx(dirn.x()), dy(dirn.y()), dz(dirn.z()),
          primary(star == Roche::PRIMARY) {}

    double operator()(double rad) const {
        // p = cofm + rad*dirn, matching Vec3 arithmetic in the original
        double px = cx + rad * dx;
        double py = rad * dy;
        double pz = rad * dz;
        double x2y2 = px * px + py * py;
        double r1sq = x2y2 + pz * pz;
        double r1 = sqrt(r1sq);
        double r2 = sqrt(r1sq + 1. - 2. * px);
        if (primary) {
            // as rpot1
            return -comp / r1 - mu / r2 - spin_sq * x2y2 / 2. + mu * px;
        }
        // as rpot2
        return -comp / r1 - mu / r2 - spin_sq * (0.5 + 0.5 * x2y2 - px) - comp * px;
    }
};

} // unnamed namespace

void Roche::face(double q, STAR star, double spin, const Subs::Vec3& dirn, double rref, double pref, double acc,
                 Subs::Vec3& pvec, Subs::Vec3& dvec, double& r, double& g){

    // centre of mass in question
    const Subs::Vec3 cofm = (star == PRIMARY) ? Subs::Vec3(0.,0.,0.) : Subs::Vec3(1.,0.,0.);

    // Potential along the ray, constants hoisted
    const RayPot rp(q, star, spin, dirn);

    // Gravity direction at the solution (one call, keep the generic version)
    Subs::Vec3 (*drp)(double q, double spin, const Subs::Vec3& p) = (star == PRIMARY) ? &drpot1 : &drpot2;

    // A check on the reference radius & potential
    double tref = rp(rref);
    if(tref < pref)
        throw Roche_Error("Roche::face error: point at reference radius = " + to_string(rref) +
                          " appears to have a lower potential = " + to_string(tref) +
                          " than the reference = " + to_string(pref) +
                          "\nOther params: q = " + to_string(q) +
                          ", direction = (" + to_string(dirn.x()) +
                          "," + to_string(dirn.y()) + "," +
                          to_string(dirn.z()) + ")" );

    // Find r1 r2 such that r1 is below reference potential and r2 is above.
    double r1 = rref/2, r2 = rref;
    tref = pref + 1;

    const int MAXSEARCH = 30;
    for(int i=0; i<MAXSEARCH && tref > pref; i++){
        r1   = r2/2;
        tref = rp(r1);
        if(tref > pref)
            r2 = r1;
    }
    if(tref > pref)
        throw Roche_Error("Roche::face error: could not find a radius with a potential below the reference potential; probably bad inputs.");

    // OK now refine with a binary chop. Crude but robust.
    const int MAXCHOP = 100;
    int nchop = 0;
    while(r2-r1 > acc && nchop < MAXCHOP){
        r = (r1+r2)/2.;
        if(rp(r) < pref){
            r1 = r;
        }else{
            r2 = r;
        }
        nchop++;
    }
    if(nchop == MAXCHOP)
        throw Roche_Error("Roche::face error: reached maximum number of binary chops = " + to_string(MAXCHOP) );

    r     = (r1+r2)/2.;
    pvec  = cofm + r*dirn;
    dvec  = drp(q, spin, pvec);
    g     = dvec.length();
    dvec /= g;
}
