#include <cmath>
#include <cstdlib>
#include <iostream>
#include "../new_subs.h"
#include "roche.h"

/**
 * rpot computes the Roche potential at a given point. This is for the standard synchronised Roche geometry
 * \param q mass ratio = M2/M1
 * \param p the point in question (units scaled by separation)
 * \return the Roche potential.
 */

double Roche::rpot(double q, const Subs::Vec3& p){

  double mu, comp, x2y2, z2, r1, r2, r1sq;

  if(q <= 0.) throw Roche_Error("q = " + to_string(q) + "(<= 0.) in rpot");

  mu   = q/(1+q);
  comp = 1.-mu;
  x2y2 = Subs::sqr(p.x()) + Subs::sqr(p.y());
  z2   = Subs::sqr(p.z());
  r1   = sqrt(r1sq = x2y2+z2);
  r2   = sqrt(r1sq + 1. - 2.*p.x());
  return (-comp/r1-mu/r2-(x2y2+mu*(mu-2.*p.x()))/2);
}

/**
 * rpot1 computes the Roche potential at a given point allowing for non-synchronous rotation of the primary.
 * \param q mass ratio = M2/M1
 * \paran spin ratio spin/orbital frequencies
 * \param p the point in question (units scaled by separation)
 * \return the Roche potential.
 */

double Roche::rpot1(double q, double spin, const Subs::Vec3& p){

  double mu, comp, x2y2, z2, r1, r2, r1sq;

  if(q <= 0.) throw Roche_Error("q = " + to_string(q) + "(<= 0.) in rpot1");

  mu   = q/(1+q);
  comp = 1.-mu;
  x2y2 = Subs::sqr(p.x()) + Subs::sqr(p.y());
  z2   = Subs::sqr(p.z());
  r1sq = x2y2 + z2;
  r1 = sqrt(r1sq);
  r2   = sqrt(r1sq + 1. - 2.*p.x());
  return (-comp/r1-mu/r2-Subs::sqr(spin)*x2y2/2.+mu*p.x());
}

/**
 * rpot2 computes the Roche potential at a given point allowing for non-synchronous rotation of the secondary.
 * \param q mass ratio = M2/M1
 * \paran spin ratio spin/orbital frequencies
 * \param p the point in question (units scaled by separation)
 * \return the Roche potential.
 */

double Roche::rpot2(double q, double spin, const Subs::Vec3& p) {
    //if(q <= 0.) throw Roche_Error("q = " + to_string(q) + "(<= 0.) in rpot2");
    
    const double mu = q/(1+q);
    const double comp = 1.-mu;
    const double px = p.x();
    const double py = p.y();
    const double pz = p.z();
    
    const double px_sq = px * px;
    const double py_sq = py * py;
    const double pz_sq = pz * pz;
    const double x2y2 = px_sq + py_sq;
    const double r1sq = x2y2 + pz_sq;
    const double r1 = sqrt(r1sq);
    
    // Optimize r2 calculation: sqrt(r1sq + 1 - 2*px)
    const double r2_sq = r1sq + 1.0 - 2.0*px;
    const double r2 = sqrt(r2_sq);
    
    // Pre-compute spin squared once
    const double spin_sq = spin * spin;
    
    // Combine terms efficiently
    return -comp/r1 - mu/r2 - spin_sq*(0.5 + 0.5*x2y2 - px) - comp*px;
}
