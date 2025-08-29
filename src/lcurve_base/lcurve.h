// AUTHOR: TOM MARSH

#ifndef LCURVE_H
#define LCURVE_H

#include <cmath>
#include <string>
#include <vector>
#include "../new_subs.h"
#include "array1d.h"
#include "../lroche_base/roche.h"
#include "nlohmann/json.hpp"



//! Lcurve namespace. Stuff to do with light curve modelling.


namespace Lcurve {
    using namespace std;
    using json = nlohmann::json;
    enum Star { STAR1, STAR2 };

    //! Name of environment variable containing name of defaults directory
    const string LCURVE_ENV = "LCURVE_ENV";

    //! Default name of defaults directory
    const string LCURVE_DIR = ".lcurve";

    /** Lcurve::Lcurve_Error is the error class for the Roche programs.
     * It is inherited from the standard string class.
     */
    class Lcurve_Error : public runtime_error {
    public:
        //! Default constructor
        Lcurve_Error() : runtime_error("Lcurve error") {
        }

        //! Constructor storing a message
        explicit Lcurve_Error(const string &err) : runtime_error(err) {
        }
    };

    //! Structure defining a single element
    /** This defines the position, area, direction, gravity and brightness of an element
     * and also any phases during which it is eclipsed.
     */
    struct Point {
        typedef vector<pair<double, double> > etype;

        //! Default constructor
        Point() : posn(), dirn(), area(0.), gravity(1.), eclipse(), flux(0.) {
        }

        //! Constructor
        Point(const Subs::Vec3 &posn_, const Subs::Vec3 &dirn_, double area_, double gravity_,
              const etype &eclipse) : posn(posn_), dirn(dirn_), area(area_), gravity(gravity_), eclipse(eclipse),
                                      flux(0.) {
        }

        //! Position vector of element (units of binary separation)
        Subs::Vec3 posn;

        //! Outward facing direction of element (unit vector)
        Subs::Vec3 dirn;

        //! Area of element (units of binary separation**2)
        float area;

        //! Gravity of element
        float gravity;

        //! Ingress and egress phases of eclipses, if any
        etype eclipse;

        //! Brightness * area
        float flux;

        //! Computes whether a point is visible (else eclipsed)
        bool visible(double phase) const {
            double phi = phase - floor(phase);
            for (size_t i = 0; i < eclipse.size(); i++) {
                const pair<double, double> p = eclipse[i];
                if ((phi >= p.first && phi <= p.second) || phi <= p.second - 1.0)
                    return false;
            }
            return true;
        }
    };

    struct Ginterp {
        //! Start phase of coarse grid 0 -- 0.5
        double phase1;

        //! End phase of coarse grid 0 -- 0.5
        double phase2;

        //! Scale factor star 1 at phase1
        double scale11;

        //! Scale factor star 1 at 1-phase1
        double scale12;

        //! Scale factor star 2 at -phase2
        double scale21;

        //! Scale factor star 2 at phase2
        double scale22;

        //! Returns scale factor for star 1 at a given phase
        double scale1(double phase) const {
            // assume coarse grid outside -phase1 to + phase1
            double pnorm = phase - floor(phase);
            if (pnorm <= phase1 || pnorm >= 1. - phase1) {
                return 1.;
            } else {
                return (scale11 * (1. - phase1 - pnorm) + scale12 * (pnorm - phase1)) / (1 - 2. * phase1);
            }
        }

        //! Returns scale factor for star 2 at a given phase
        double scale2(double phase) const {
            double pnorm = phase - floor(phase);
            if (pnorm >= phase2 && pnorm <= 1. - phase2) {
                return 1.;
            } else if (pnorm < 0.5) {
                return (scale22 * (phase2 - pnorm) + scale21 * (pnorm + phase2)) / (2. * phase2);
            } else {
                return (scale21 * (1. + phase2 - pnorm) + scale22 * (pnorm - 1. + phase2)) / (2. * phase2);
            }
        }

        //! Returns integer type to represent situation we are in
        int type(double phase) const {
            double pnorm = phase - floor(phase);
            if (pnorm <= phase1 || pnorm >= 1. - phase1) {
                // coarse grid for star 2, fine for star 1
                return 1;
            } else if ((pnorm > phase1 && pnorm < phase2) || (pnorm > 1. - phase2 && pnorm < 1. - phase1)) {
                // coarse grid for both stars
                return 2;
            } else {
                // coarse grid for star 1, fine for star 2
                return 3;
            }
        }
    };

    istream &operator>>(istream &s, Point &p);

    ostream &operator<<(ostream &s, const Point &p);

    //! Physical parameter structure
    /** Holds basic information for a physical parameter which are its value,
     * range for varying it, step size for derivative computation, whether it
     * is variable or not, and, optionally, whether it is defined or not
     */
    struct Pparam {
        //! Default constructor
        Pparam() : value(0), range(0), dstep(0), vary(false), defined(false) {
        }

        //! Constructor from a string
        /** Sets the values from a string of the form "0.12405 0.001"
         * giving the value and the step size. 0 is interpreted as holding
         * a parameter fixed.
         */
        Pparam(const string &entry) {
            istringstream istr(entry);
            istr >> value >> range >> dstep >> vary;
            if (!istr)
                throw Lcurve::Lcurve_Error(string("Pparam: could not read entry = ") + entry);
            istr >> defined;
            if (!istr) defined = true;
        }

        //! Implicit conversion to a double by just returning the value
        operator double() const { return value; }

        //! The value of the parameter
        double value;

        //! The value of the range over which to vary it
        double range;

        //! The value of the step size for derivative computation
        double dstep;

        //! Whether the parameter varies
        bool vary;

        //! Whether the parameter has been defined
        bool defined;
    };


    //! ASCII input operator for a physical parameter
    ostream &operator<<(ostream &s, const Pparam &p);

    //! Lim darkening class
    class LDC {
    public:
        enum LDCtype { POLY, CLARET };

        //! Default. Sets all to zero.
        LDC() : ldc1(0.), ldc2(0.), ldc3(0.), ldc4(0.), mucrit(0.), ltype(POLY) {
        }

        //! Standard constructor
        LDC(double ldc1, double ldc2, double ldc3, double ldc4, double mucrit, LDCtype ltype) : ldc1(ldc1), ldc2(ldc2),
            ldc3(ldc3), ldc4(ldc4), mucrit(mucrit), ltype(ltype) {
        }

        //! Computes I(mu)
        double imu(double mu) const {
            if (mu <= 0) {
                return 0.;
            } else {
                mu = min(mu, 1.);
                double ommu = 1. - mu, im = 1.;
                if (this->ltype == POLY) {
                    im -= ommu * (this->ldc1 + ommu * (this->ldc2 + ommu * (this->ldc3 + ommu * this->ldc4)));
                } else if (this->ltype == CLARET) {
                    im -= this->ldc1 + this->ldc2 + this->ldc3 + this->ldc4;
                    double msq = sqrt(mu);
                    im += msq * (this->ldc1 + msq * (this->ldc2 + msq * (this->ldc3 + msq * this->ldc4)));
                }
                return im;
            }
        }

        //! To help applying mucrit
        bool see(double mu) const { return mu > this->mucrit; }

    private:
        double ldc1;
        double ldc2;
        double ldc3;
        double ldc4;
        double mucrit;
        LDCtype ltype;
    };

    //! Model structure
    /** Defines the model to be used and which parameters are to be
     * varied in the fit.  The order of the paremeters here defines the
     * order in which they must occur in any parameter vector to be fed
     * to a generic minimisation routine such as amoeba.  Star 1 is
     * taken to be spherical. Star 2 can be tidally distorted.
     */
    //! Model structure
    struct Model {
        // Constructors and public methods
        Model(const json& config);

        int nvary() const;

        void set_param(const Subs::Array1D<double> &vpar);

        bool is_not_legal(const Subs::Array1D<double> &vpar) const;

        Subs::Array1D<double> get_param() const;

        Subs::Array1D<double> get_range() const;

        Subs::Array1D<double> get_dstep() const;

        vector<pair<double, double>> get_limit() const;

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

    //! ASCII output operator for a Model
    ostream &operator<<(ostream &s, const Model &model);

    //! Holds all the data for a single point of a light curve
    struct Datum {
        //! The time
        double time;

        //! The exposure length in the same units as the time
        double expose;

        //! The flux
        double flux;

        //! The uncertainty on the flux in the same units
        double ferr;

        //! Weight factor for calculating goodness of fit
        double weight;

        //! Factor to split up data points to allow for finite exposures
        int ndiv;
    };

    //! ASCII input of a Datum (expects time expose flux ferr weight ndiv)
    istream &operator>>(istream &s, Datum &datum);

    //! ASCII output of a Datum (expects time expose flux ferr weight ndiv)
    ostream &operator<<(ostream &s, const Datum &datum);

    // Holds a light curve
    class Data : public vector<Datum> {
    public:
        //! Default constructor
        Data() : vector<Datum>() {
        }

        //! Constructor with pre-defined size
        Data(int n) : vector<Datum>(n) {
        }

        //! Constructor from a file
        Data(const string &file);

        //! Writes to an ASCII file
        void wrasc(const string &file) const;

        //! Reads from an ASCII file
        void rasc(const string &file);
    };


    //! Function object to compute chi**2

    /** This is inherited from the abstract class Subs::Afunc to define the basic
     * functionality needed for it to be used inside minimisation routines like amoeba
     */

    class Fobj : public Subs::Afunc {
    public:
        //! Total number of calls to Lcurve::chisq
        static int neval;

        //! Minimum chi**2 ever encountered
        static double chisq_min;

        //! Scale factors for minimum chi**2
        static vector<double> scale_min;

        //! Constructor; stores the data needed to compute chi**2
        Fobj(const Model &model, const Data &data) : model(model), data(data) {
        }

        //! Function call operator (overload)
        double operator()(const Subs::Array1D<double> &vpar);

        //! Computes chi-squared
        double chisq();

        //! Return fit values
        double operator[](int n) const;

    private:
        Model model;
        const Data &data;
        Subs::Array1D<double> fit;
    };

    //! Computes position and direction on a disc
    void pos_disc(double r, double theta, double beta, double height, Subs::Vec3 &posn, Subs::Vec3 &dirn);

    //! Computes the number of faces given the number of latitude strips to be used
    int numface(int nlat, bool infill, double thelo, double thehi, int nlatfill, int nlngfill);

    //! Computes elements over the primary or secondary star
    void set_star_grid(const Model &mdl, Roche::STAR which_star, bool fine, vector<Lcurve::Point> &star);

    void add_faces(vector<Lcurve::Point> &star, int &nface, double tlo, double thi, double dtheta, int nlatfill,
                   int nlngfill,
                   bool npole, Roche::STAR which_star, double q, double iangle, double r1, double r2, double rref1,
                   double rref2,
                   bool roche1, bool roche2, double spin1, double spin2, bool eclipse, double gref, double pref1,
                   double pref2,
                   double ffac1, double ffac2, double delta);

    //! Computes elements over a disc
    void set_disc_grid(const Model &mdl, vector<Lcurve::Point> &disc);

    //! Bright spot elelemts
    void set_bright_spot_grid(const Model &mdl, vector<Lcurve::Point> &spot);

    //! Computes elements at rim of disc
    void set_disc_edge(const Model &mdl, bool outer,
                       vector<Lcurve::Point> &edge,
                       bool visual = true);

    //! Sets the continuum element contributions
    void set_star_continuum(const Model &mdl,
                            vector<Lcurve::Point> &star1,
                            vector<Lcurve::Point> &star2);

    //! Sets the disc continuum brightness.
    void set_disc_continuum(double rdisc, double tdisc, double texp,
                            double wave, vector<Lcurve::Point> &disc);

    //! Sets the disc edge continuum brightness.
    void set_edge_continuum(double tedge, double r2, double t2,
                            double absorb, double wave,
                            vector<Lcurve::Point> &edge);

    //! Sets the emission line brightness
    void set_star_emission(double limb2, double hbyr,
                           const vector<Lcurve::Point> &star2,
                           vector<float> &bright2);

    //! Computes the flux at a given phase
    double comp_light(double iangle, const LDC &ldc1, const LDC &ldc2,
                      double lin_limb_disc, double quad_limb_disc,
                      double phase, double expose, int ndiv, double q,
                      double beam_factor1, double beam_factor2,
                      double spin1, double spin2, float vscale, bool glens1,
                      double rlens1, const Ginterp &gint,
                      const vector<Lcurve::Point> &star1f,
                      const vector<Lcurve::Point> &star2f,
                      const vector<Lcurve::Point> &star1c,
                      const vector<Lcurve::Point> &star2c,
                      const vector<Lcurve::Point> &disc,
                      const vector<Lcurve::Point> &edge,
                      const vector<Lcurve::Point> &spot);

    //! Computes flux from star 1 only
    double comp_star1(double iangle, const LDC &ldc1, double phase,
                      double expose, int ndiv, double q, double beam_factor1,
                      float vscale, const Ginterp &gint,
                      const vector<Lcurve::Point> &star1f,
                      const vector<Lcurve::Point> &star1c);

    //! Computes flux from star 2 only
    double comp_star2(double iangle, const LDC &ldc2, double phase,
                      double expose, int ndiv, double q, double beam_factor2,
                      float vscale, bool glens1, double rlens1,
                      const Ginterp &gint,
                      const vector<Lcurve::Point> &star2f,
                      const vector<Lcurve::Point> &star2c);

    //! Computes flux from disc
    double comp_disc(double iangle, double lin_limb_disc, double quad_limb_disc,
                     double phase, double expose, int ndiv, double q,
                    const vector<Lcurve::Point> &disc);

    //! Computes flux from disc edge
    double comp_edge(double iangle, double lin_limb_disc, double quad_limb_disc,
                     double phase, double expose, int ndiv, double q,
                     const vector<Lcurve::Point> &edge);

    //! Compute flux from spot
    double comp_spot(double iangle, double phase, double expose, int ndiv,
                     double q,
                     const vector<Lcurve::Point> &spot);

    //! Compute flux-weighted gravity of star 1
    double comp_gravity1(const Model &mdl,
                         const vector<Lcurve::Point> &star1);

    //! Compute flux-weighted gravity of star 2
    double comp_gravity2(const Model &mdl,
                         const vector<Lcurve::Point> &star2);

    //! Compute volume-averaged radius of star 1
    double comp_radius1(const vector<Lcurve::Point> &star1);

    //! Compute volume-averaged radius of star 2
    double comp_radius2(const vector<Lcurve::Point> &star2);

    //! Convenience routine
    void star_eclipse(double q, double r, double spin, double ffac,
                      double iangle, const Subs::Vec3 &posn, double delta,
                      bool roche, Roche::STAR star, Point::etype &eclipses);

    //! Computes eclipse by a flared disc
    bool disc_eclipse(double iangle, double phase, double rdisc1,
                      double rdisc2, double beta, double height,
                      const Subs::Vec3 &posn);

    //! Computes eclipse by a flared disc for points
    bool disc_surface_eclipse(double iangle, double phase, double rdisc1,
                              double rdisc2, double beta, double height,
                              const Subs::Vec3 &posn);

    //! Computes an entire light curve corresponding to a given data set.
    void light_curve_comp(const Lcurve::Model &model, const Lcurve::Data &data,
                          bool scale, bool rdata, bool info, vector<double> &sfac,
                          vector<double> &calc, double &wdwarf,
                          double &chisq, double &wnok,
                          double &logg1, double &logg2, double &rv1, double &rv2);

    //! Re-scales a fit to minimise chi**2
    double re_scale(const Lcurve::Data &data, vector<double> &fit,
                    double &chisq, double &wnok);
};

#endif //LCURVE_H
