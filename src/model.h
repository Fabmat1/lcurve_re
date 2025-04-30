// Model and pparam definitions to closely follow what lcurve does

#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include "lcurve_base/array1d.h"
#include "ldc.h"
#include <nlohmann/json.hpp>
using json = nlohmann::json;

using namespace std;


struct Pparam {
    double value{0.0};
    double range{0.0};
    double dstep{0.0};
    bool vary{false};
    bool defined{false};

    // Default constructor
    Pparam() = default;

    // Constructor from a string
    explicit Pparam(const string &entry) {
        istringstream istr(entry);
        if (!(istr >> value >> range >> dstep >> vary >> defined)) {
            throw runtime_error(
                "Pparam: too little values in entry (need to be 'value range dstep vary defined') = " + entry);
        }

        // Now check: are there *extra* values after the 5?
        string leftover;
        if (istr >> leftover) {
            throw runtime_error(
                "Pparam: too many values in entry = (need to be 'value range dstep vary defined')" + entry);
        }
    }

    // Implicit conversion to double
    operator double() const noexcept { return value; }
};


//! Model structure
/**
 * Defines the model to be used and which parameters are to be
 * varied in the fit. The order of the parameters defines the order
 * expected by minimisation routines (e.g., amoeba). Star 1 is
 * assumed spherical; Star 2 can be tidally distorted.
 */
struct Model {
    // Constructors and public methods
    Model(json config);

    int nvary() const;

    void set_param(const Subs::Array1D<double> &vpar);

    bool is_not_legal(const Subs::Array1D<double> &vpar) const;

    Subs::Array1D<double> get_param() const;

    Subs::Array1D<double> get_range() const;

    Subs::Array1D<double> get_dstep() const;

    void wrasc(const string &file) const;

    string get_name(int i) const;

    // Physical parameters (can be varied)
    Pparam q, iangle, r1, r2;
    Pparam cphi3, cphi4, spin1, spin2;
    Pparam t1, t2;
    Pparam ldc1_1, ldc1_2, ldc1_3, ldc1_4;
    Pparam ldc2_1, ldc2_2, ldc2_3, ldc2_4;
    Pparam velocity_scale, beam_factor1, beam_factor2;
    Pparam t0, period, pdot, deltat;
    Pparam gravity_dark1, gravity_dark2;
    Pparam absorb, slope, quad, cube, third;

    // Disc parameters
    Pparam rdisc1, rdisc2, height_disc, beta_disc;
    Pparam temp_disc, texp_disc;
    Pparam lin_limb_disc, quad_limb_disc;
    Pparam temp_edge, absorb_edge;

    // Bright spot parameters
    Pparam radius_spot, length_spot, height_spot;
    Pparam expon_spot, epow_spot, angle_spot, yaw_spot;
    Pparam temp_spot, tilt_spot, cfrac_spot;

    // Star spots (1st and 2nd star)
    Pparam stsp11_long, stsp11_lat, stsp11_fwhm, stsp11_tcen;
    Pparam stsp12_long, stsp12_lat, stsp12_fwhm, stsp12_tcen;
    Pparam stsp13_long, stsp13_lat, stsp13_fwhm, stsp13_tcen;
    Pparam stsp21_long, stsp21_lat, stsp21_fwhm, stsp21_tcen;
    Pparam stsp22_long, stsp22_lat, stsp22_fwhm, stsp22_tcen;

    // Uneclipsed (UE) spot parameters
    Pparam uesp_long1, uesp_long2, uesp_lathw;
    Pparam uesp_taper, uesp_temp;

    // Computational parameters
    double delta_phase; //!< Phase accuracy for Roche computations
    int nlat1f, nlat2f; //!< Fine grid latitudinal strips
    int nlat1c, nlat2c; //!< Coarse grid latitudinal strips
    bool npole; //!< Use genuine pole rather than substellar point
    int nlatfill, nlngfill; //!< Extra sampling along track
    double lfudge, llo, lhi; //!< Fine strip latitudes
    double phase1, phase2; //!< Coarse grid phase range
    double wavelength; //!< Wavelength (microns?)
    bool roche1, roche2; //!< Account for Roche distortions
    bool eclipse1, eclipse2; //!< Eclipses
    bool glens1; //!< Gravitational lensing by star 1
    bool use_radii; //!< Use radii vs. contact phases
    double tperiod; //!< True period (days)
    bool gdark_bolom1, gdark_bolom2; //!< Bolometric gravity darkening
    double mucrit1, mucrit2; //!< Critical mu values for limb darkening

    LDC::LDCtype limb1, limb2; //!< Limb darkening types
    bool mirror; //!< Mirror reflection model
    bool add_disc; //!< Add disc?
    int nrad; //!< Radial disc strips
    bool opaque; //!< Disc opacity
    bool add_spot; //!< Add bright spot?
    int nspot; //!< Number of spot elements
    bool iscale; //!< Individual component scaling

    //! Returns relative radii accounting for parameterisation method
    void get_r1r2(double &rr1, double &rr2) const {
        if (use_radii) {
            rr1 = r1;
            rr2 = r2;
        } else {
            double sini = sin(Subs::deg2rad(iangle));
            double r2pr1 = sqrt(1.0 - pow(sini * cos(2.0 * M_PI * cphi4.value), 2));
            double r2mr1 = sqrt(1.0 - pow(sini * cos(2.0 * M_PI * cphi3.value), 2));
            rr1 = (r2pr1 - r2mr1) / 2.0;
            rr2 = (r2pr1 + r2mr1) / 2.0;
        }
    }

    //! Returns limb darkening struct for star 1
    LDC get_ldc1() const {
        return {ldc1_1, ldc1_2, ldc1_3, ldc1_4, mucrit1, limb1};
    }

    //! Returns limb darkening struct for star 2
    LDC get_ldc2() const {
        return {ldc2_1, ldc2_2, ldc2_3, ldc2_4, mucrit2, limb2};
    }
};


#endif //MODEL_H
