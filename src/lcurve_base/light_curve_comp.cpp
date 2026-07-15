#include "lcurve.h"
#include "constants.h"
#include <iostream>

/** This routine computes the light curve corresponding to a particular
 * model and times defined by some data. Data with negative or zero errors
 * are skipped for speed, corresponding fit values are set = 0.
 * \param mdl   the model
 * \param data  the data defining the times
 * \param scale work out linear scale factors by svd or not. This will either be a single number for all components
 *              or a different one for each, depending upon the modle parameter iscale.
 * \param rdata true if real data (then chi**2 will be computed)
 * \param info  if true, it prints out array sizes to stderr
 * \param sfac  scaling factors of each of the components star1, star2, disc spot. Will be determined by
 *              svd if scale=true, otherwise values on entry are used. Only sfac[0] will be used if
 *              model parameter iscale=no.
 * \param calc  the computed light curve
 * \param wdwarf contri<tion of the white at phase 0.5
 * \param chisq  chi**2 value
 * \param wnok   weighted number of data points
 * \param logg2  flux-weighted logg for star 2, CGS units
 */

using namespace std;

namespace {

// Exact-reuse cache for the star grids across successive evaluations.
// Keys hold every model parameter the cached objects depend on; a cache
// hit therefore requires identical inputs and reproduces identical
// outputs (this matters for finite-difference Jacobians, where most
// parameter steps do not touch the geometry).

struct GeomKey {
    double q, iangle, r1, r2, cphi3, cphi4, spin1, spin2;
    double velocity_scale, tperiod, llo, lhi, lfudge, delta_phase;
    double phase1, phase2;
    int nlat1f, nlat2f, nlat1c, nlat2c, nlatfill, nlngfill;
    bool use_radii, roche1, roche2, eclipse1, eclipse2, glens1, npole;
    bool operator==(const GeomKey &) const = default;
};

struct ContKey {
    double t1, t2, gd1, gd2, wavelength, absorb;
    double l11, l12, l13, l14, l21, l22, l23, l24;
    double mucrit1, mucrit2, beam1, beam2;
    int limb1, limb2;
    bool gdb1, gdb2, mirror;
    double sp[25];
    bool spdef[25];
    bool operator==(const ContKey &) const = default;
};

struct GridCacheTL {
    bool has_geom = false, has_cont = false;
    GeomKey gkey{};
    ContKey ckey{};
    bool copy1 = true, copy2 = true;
    std::vector<Lcurve::Point> star1f, star2f, star1c_own, star2c_own;
    Lcurve::Ginterp gint{};
    Lcurve::FlatGrid fg1f, fg2f, fg1c_own, fg2c_own;
};

GeomKey make_geom_key(const Lcurve::Model &m) {
    GeomKey k{};
    k.q = m.q.value; k.iangle = m.iangle.value;
    k.r1 = m.r1.value; k.r2 = m.r2.value;
    k.cphi3 = m.cphi3.value; k.cphi4 = m.cphi4.value;
    k.spin1 = m.spin1.value; k.spin2 = m.spin2.value;
    // velocity_scale/tperiod only shape the grids through the lensing
    // correction; without lensing the geometry is independent of them,
    // so key them out to keep cache hits on e.g. velocity_scale steps.
    k.velocity_scale = m.glens1 ? m.velocity_scale.value : 0.;
    k.tperiod = m.glens1 ? m.tperiod : 0.;
    k.llo = m.llo; k.lhi = m.lhi; k.lfudge = m.lfudge;
    k.delta_phase = m.delta_phase;
    k.phase1 = m.phase1; k.phase2 = m.phase2;
    k.nlat1f = m.nlat1f; k.nlat2f = m.nlat2f;
    k.nlat1c = m.nlat1c; k.nlat2c = m.nlat2c;
    k.nlatfill = m.nlatfill; k.nlngfill = m.nlngfill;
    k.use_radii = m.use_radii;
    k.roche1 = m.roche1; k.roche2 = m.roche2;
    k.eclipse1 = m.eclipse1; k.eclipse2 = m.eclipse2;
    k.glens1 = m.glens1; k.npole = m.npole;
    return k;
}

ContKey make_cont_key(const Lcurve::Model &m) {
    ContKey k{};
    k.t1 = m.t1.value; k.t2 = m.t2.value;
    k.gd1 = m.gravity_dark1.value; k.gd2 = m.gravity_dark2.value;
    k.wavelength = m.wavelength; k.absorb = m.absorb.value;
    k.l11 = m.ldc1_1.value; k.l12 = m.ldc1_2.value;
    k.l13 = m.ldc1_3.value; k.l14 = m.ldc1_4.value;
    k.l21 = m.ldc2_1.value; k.l22 = m.ldc2_2.value;
    k.l23 = m.ldc2_3.value; k.l24 = m.ldc2_4.value;
    k.mucrit1 = m.mucrit1; k.mucrit2 = m.mucrit2;
    k.beam1 = m.beam_factor1.value; k.beam2 = m.beam_factor2.value;
    k.limb1 = int(m.limb1); k.limb2 = int(m.limb2);
    k.gdb1 = m.gdark_bolom1; k.gdb2 = m.gdark_bolom2;
    k.mirror = m.mirror;
    const Lcurve::Pparam *sp[25] = {
        &m.stsp11_long, &m.stsp11_lat, &m.stsp11_fwhm, &m.stsp11_tcen,
        &m.stsp12_long, &m.stsp12_lat, &m.stsp12_fwhm, &m.stsp12_tcen,
        &m.stsp13_long, &m.stsp13_lat, &m.stsp13_fwhm, &m.stsp13_tcen,
        &m.stsp21_long, &m.stsp21_lat, &m.stsp21_fwhm, &m.stsp21_tcen,
        &m.stsp22_long, &m.stsp22_lat, &m.stsp22_fwhm, &m.stsp22_tcen,
        &m.uesp_long1, &m.uesp_long2, &m.uesp_lathw, &m.uesp_taper,
        &m.uesp_temp};
    for (int i = 0; i < 25; i++) {
        k.sp[i] = sp[i]->value;
        k.spdef[i] = sp[i]->defined;
    }
    return k;
}

} // unnamed namespace

void Lcurve::light_curve_comp(const Lcurve::Model &mdl,
                              const Lcurve::Data &data, bool scale,
                              bool rdata, bool info,
                              vector<double> &sfac,
                              vector<double> &calc, double &wdwarf,
                              double &chisq, double &wnok,
                              double &logg1, double &logg2, double &rv1, double &rv2,
                              bool diagnostics) {
    // Get the size right
    calc.resize(static_cast<int>(data.size()));

    double r1, r2;
    mdl.get_r1r2(r1, r2);
    double rl2 = 1. - Roche::xl12(mdl.q, mdl.spin2);
    if (r2 < 0)
        r2 = rl2;
    else if (r2 > rl2){
        std::cout << "r2: " << r2 << "rl2 :" << rl2 << std::endl;
        throw Lcurve_Error("light_curve_comp: the secondary star is larger than its Roche lobe!");
    }
        

    LDC ldc1 = mdl.get_ldc1();
    LDC ldc2 = mdl.get_ldc2();

    // Compute gravitational radius of star 1 if need be. An extra factor
    // of 4 saves multiplication in the innermost loops later
    double rlens1 = 0.;
    if (mdl.glens1) {
        // Compute G(M1+M2), SI, and the separation a, SI.
        double gm = pow(1000. * mdl.velocity_scale, 3) * mdl.tperiod * Constants::DAY / Constants::TWOPI;
        double a = pow(
            gm / (Constants::TWOPI / Constants::DAY / mdl.tperiod * Constants::TWOPI / Constants::DAY / mdl.tperiod),
            1. / 3.);
        rlens1 = 4. * gm / (1. + mdl.q) / a / (Constants::C * Constants::C);
    }

    // Star grids come from a per-thread exact-reuse cache: when every
    // parameter a stage depends on is unchanged since the previous call,
    // the stage is skipped and its cached (identical) result reused.
    // Opaque discs append per-model eclipse ranges to the star grids, so
    // caching is bypassed in that case.
    static thread_local GridCacheTL tls_cache;
    static const bool cache_env_off = getenv("LCURVE_NO_GRID_CACHE") != nullptr;
    const bool use_cache = !cache_env_off && !(mdl.add_disc && mdl.opaque);

    GridCacheTL local_cache;
    GridCacheTL &C = use_cache ? tls_cache : local_cache;

    GeomKey gk = make_geom_key(mdl);
    if (!C.has_geom || !(C.gkey == gk)) {
        set_star_grid(mdl, Roche::PRIMARY, true, C.star1f);
        set_star_grid(mdl, Roche::SECONDARY, true, C.star2f);

        // When the coarse grid would be identical to the fine one, alias it
        // instead of building (or deep-copying) it.
        C.copy1 = (mdl.nlat1f == mdl.nlat1c);
        C.copy2 = (mdl.nlat2f == mdl.nlat2c) &&
                  (!mdl.npole || r1 >= r2 || (mdl.nlatfill == 0 && mdl.nlngfill == 0));
        if (!C.copy1)
            set_star_grid(mdl, Roche::PRIMARY, false, C.star1c_own);
        else
            C.star1c_own.clear();
        if (!C.copy2)
            set_star_grid(mdl, Roche::SECONDARY, false, C.star2c_own);
        else
            C.star2c_own.clear();

        C.gkey = gk;
        C.has_geom = true;
        C.has_cont = false;
    }

    const bool copy1 = C.copy1, copy2 = C.copy2;
    vector<Point> &star1f = C.star1f, &star2f = C.star2f;
    vector<Point> &star1c = copy1 ? C.star1f : C.star1c_own;
    vector<Point> &star2c = copy2 ? C.star2f : C.star2c_own;
    vector<Point> disc, edge, spot;

    if (info) {
        cerr << "Number of points for star 1 (fine) = " << star1f.size() << endl;
        cerr << "Number of points for star 2 (fine) = " << star2f.size() << endl;
        cerr << "Number of points for star 1 (coarse) = " << star1c.size() << endl;
        cerr << "Number of points for star 2 (coarse) = " << star2c.size() << endl;
    }

    ContKey ck = make_cont_key(mdl);
    if (!C.has_cont || !(C.ckey == ck)) {
        set_star_continuum(mdl, star1f, star2f);

        // Aliased grids already have their continuum set; recomputation
        // would be identical, so only run when a coarse grid is separate.
        if (!copy1 || !copy2)
            set_star_continuum(mdl, star1c, star2c);

        // Work out grid switching parameters
        C.gint = Ginterp{};
        C.gint.phase1 = mdl.phase1;
        C.gint.phase2 = mdl.phase2;
        C.gint.scale11 = C.gint.scale12 = C.gint.scale21 = C.gint.scale22 = 1.;

        if (mdl.nlat1c != mdl.nlat1f) {
            double ff = comp_star1(mdl.iangle, ldc1, 0.9999999999 * mdl.phase1,
                                   0., 1, mdl.q, mdl.beam_factor1,
                                   mdl.velocity_scale, C.gint, star1f, star1c);
            double fc = comp_star1(mdl.iangle, ldc1, 1.0000000001 * mdl.phase1,
                                   0., 1, mdl.q, mdl.beam_factor1,
                                   mdl.velocity_scale, C.gint, star1f, star1c);
            C.gint.scale11 = ff / fc;
            ff = comp_star1(mdl.iangle, ldc1, 1. - 0.9999999999 * mdl.phase1,
                            0., 1, mdl.q, mdl.beam_factor1, mdl.velocity_scale,
                            C.gint, star1f, star1c);
            fc = comp_star1(mdl.iangle, ldc1, 1. - 1.0000000001 * mdl.phase1,
                            0., 1, mdl.q, mdl.beam_factor1, mdl.velocity_scale,
                            C.gint, star1f, star1c);
            C.gint.scale12 = ff / fc;
        }

        if (!copy2) {
            double ff = comp_star2(mdl.iangle, ldc2, 1 - 1.0000000001 * mdl.phase2,
                                   0., 1, mdl.q, mdl.beam_factor2,
                                   mdl.velocity_scale, mdl.glens1, rlens1,
                                   C.gint, star2f, star2c);
            double fc = comp_star2(mdl.iangle, ldc2, 1 - 0.9999999999 * mdl.phase2,
                                   0., 1, mdl.q, mdl.beam_factor2,
                                   mdl.velocity_scale, mdl.glens1, rlens1,
                                   C.gint, star2f, star2c);
            C.gint.scale21 = ff / fc;
            ff = comp_star2(mdl.iangle, ldc2, 1.0000000001 * mdl.phase2, 0., 1,
                            mdl.q, mdl.beam_factor2, mdl.velocity_scale,
                            mdl.glens1, rlens1, C.gint, star2f, star2c);
            fc = comp_star2(mdl.iangle, ldc2, 0.9999999999 * mdl.phase2, 0., 1,
                            mdl.q, mdl.beam_factor2, mdl.velocity_scale,
                            mdl.glens1, rlens1, C.gint, star2f, star2c);
            C.gint.scale22 = ff / fc;
        }

        // Flatten the star grids into SoA form for the fast flux kernels.
        // With an opaque disc (cache bypassed) this happens later instead,
        // after the disc eclipses have been appended to the star grids.
        if (use_cache) {
            C.fg1f.build(star1f);
            C.fg2f.build(star2f);
            if (!copy1) C.fg1c_own.build(star1c);
            if (!copy2) C.fg2c_own.build(star2c);
        }

        C.ckey = ck;
        C.has_cont = true;
    }

    const Ginterp gint = C.gint;

    if (mdl.add_disc) {
        // set disc upper surface and outer edge
        Lcurve::set_disc_grid(mdl, disc);
        Lcurve::set_disc_edge(mdl, true, edge, false);

        if (info) {
            cerr << "Number of points for the disc = " << disc.size()
                 << endl;
        }

        // note that the inner radius of the disc is set equal to that of the
        // white dwarf if rdisc1 <= 0 while the outer disc is set equal to the
        // spot radius
        double rdisc1 = mdl.rdisc1 > 0. ? mdl.rdisc1 : r1;
        double rdisc2 = mdl.rdisc2 > 0. ? mdl.rdisc2 : mdl.radius_spot;

        if (mdl.opaque) {
            vector<pair<double, double> > eclipses;
            // Apply eclipse by disc to star 1
            for (long unsigned int i = 0; i < star1f.size(); i++) {
                eclipses = Roche::disc_eclipse(mdl.iangle, rdisc1, rdisc2,
                                               mdl.beta_disc, mdl.height_disc,
                                               star1f[i].posn);
                for (size_t j = 0; j < eclipses.size(); j++)
                    star1f[i].eclipse.push_back(eclipses[j]);
            }
            // skip when star1c aliases star1f (already appended above)
            for (long unsigned int i = 0; !copy1 && i < star1c.size(); i++) {
                eclipses = Roche::disc_eclipse(mdl.iangle, rdisc1, rdisc2,
                                               mdl.beta_disc, mdl.height_disc,
                                               star1c[i].posn);
                for (size_t j = 0; j < eclipses.size(); j++)
                    star1c[i].eclipse.push_back(eclipses[j]);
            }

            // Apply eclipse by disc to star 2
            for (long unsigned int i = 0; i < star2f.size(); i++) {
                eclipses = Roche::disc_eclipse(mdl.iangle, rdisc1, rdisc2,
                                               mdl.beta_disc, mdl.height_disc,
                                               star2f[i].posn);
                for (size_t j = 0; j < eclipses.size(); j++)
                    star2f[i].eclipse.push_back(eclipses[j]);
            }
            // skip when star2c aliases star2f (already appended above)
            for (long unsigned int i = 0; !copy2 && i < star2c.size(); i++) {
                eclipses = Roche::disc_eclipse(mdl.iangle, rdisc1, rdisc2,
                                               mdl.beta_disc, mdl.height_disc,
                                               star2c[i].posn);
                for (size_t j = 0; j < eclipses.size(); j++)
                    star2c[i].eclipse.push_back(eclipses[j]);
            }
        }

        // Set the surface brightness of the disc
        set_disc_continuum(rdisc2, mdl.temp_disc, mdl.texp_disc,
                           mdl.wavelength, disc);

        // Set the surface brightness of outer edge, accounting for
        // irradiation by star 2
        set_edge_continuum(mdl.temp_edge, r2, abs(mdl.t2),
                           mdl.absorb_edge, mdl.wavelength, edge);
    }

    // This could raise an exception for bad parameters.
    if (mdl.add_spot) Lcurve::set_bright_spot_grid(mdl, spot);

    // With an opaque disc the cache is bypassed and the flat grids are
    // built here, after the disc-eclipse appends above.
    if (!use_cache) {
        C.fg1f.build(star1f);
        C.fg2f.build(star2f);
        if (!copy1) C.fg1c_own.build(star1c);
        if (!copy2) C.fg2c_own.build(star2c);
    }
    const FlatGrid &fg1f = C.fg1f, &fg2f = C.fg2f;
    const FlatGrid &fg1c = copy1 ? C.fg1f : C.fg1c_own;
    const FlatGrid &fg2c = copy2 ? C.fg2f : C.fg2c_own;

    // polynomial fudge factor stuff: slope, quad, cube
    double xmin = data[0].time, xmax = data[0].time;
    for (size_t np = 1; np < data.size(); np++) {
        xmin = data[np].time > xmin ? xmin : data[np].time;
        xmax = data[np].time < xmax ? xmax : data[np].time;
    }
    double middle = (xmin + xmax) / 2., range = (xmax - xmin) / 2.;

    // Compute light curve. Per-point phases first (cheap, serial).
    vector<double> phases(data.size()), slfacs(data.size());
    for (size_t np = 0; np < data.size(); np++) {
        // Compute phase, accounting for quadratic term
        double phase = (data[np].time - mdl.t0) / mdl.period;

        // small Newton-Raphson iteration
        for (int it = 0; it < 4; it++) {
            phase -= (mdl.t0 + phase * (mdl.period + mdl.pdot * phase) - data[np].time) /
                    (mdl.period + 2. * mdl.pdot * phase);
        }

        // advance/retard by time offset between primary & secondary eclipse
        phase += mdl.deltat / mdl.period / 2. * (cos(2. * Constants::PI * phase) - 1.);

        phases[np] = phase;
        double frac = (data[np].time - middle) / range;
        slfacs[np] = 1. + frac * (mdl.slope + frac * (mdl.quad + frac * mdl.cube));
    }

    vector<vector<double>> fcomp;
    if (mdl.iscale) {
        // Individual component scaling: per-point evaluation with the
        // single-phase kernels (components must stay separate).
        fcomp.assign(data.size(), vector<double>(mdl.t2 > 0 ? 5 : 4));

        #pragma omp parallel for schedule(dynamic,1)
        for (size_t np = 0; np < data.size(); np++) {
            double phase = phases[np], slfac = slfacs[np];
            double expose = data[np].expose / mdl.period;

            fcomp[np][0] = slfac * comp_star1_flat(mdl.iangle, ldc1, phase, expose,
                                              data[np].ndiv, mdl.q,
                                              mdl.beam_factor1, mdl.velocity_scale,
                                              gint, fg1f, fg1c);

            fcomp[np][1] = slfac * comp_disc(mdl.iangle, mdl.lin_limb_disc,
                                             mdl.quad_limb_disc, phase, expose,
                                             data[np].ndiv, mdl.q,
                                             disc);

            fcomp[np][2] = slfac * comp_edge(mdl.iangle, mdl.lin_limb_disc,
                                             mdl.quad_limb_disc, phase, expose,
                                             data[np].ndiv, mdl.q,
                                             edge);

            fcomp[np][3] = slfac * comp_spot(mdl.iangle, phase, expose,
                                             data[np].ndiv, mdl.q,
                                             spot);

            if (mdl.t2 > 0)
                fcomp[np][4] = slfac * comp_star2_flat(mdl.iangle, ldc2, phase, expose,
                                                  data[np].ndiv, mdl.q,
                                                  mdl.beam_factor2,
                                                  mdl.velocity_scale,
                                                  mdl.glens1, rlens1,
                                                  gint, fg2f, fg2c);
        }
    } else {
        // Batched path: collect every sub-exposure phase, then sweep each
        // grid once (blockwise, all phases per block) for cache locality.
        const double ri = Subs::deg2rad(mdl.iangle);
        const double cosi = cos(ri), sini = sin(ri);
        float vs = mdl.velocity_scale;
        const double XCOFM = mdl.q / (1.0 + mdl.q);
        const double VFAC = vs / (Constants::C / 1.e3);

        PhaseBatch pb1f, pb1c, pb2f, pb2c;
        for (size_t np = 0; np < data.size(); np++) {
            double phase = phases[np], slfac = slfacs[np];
            double expose = data[np].expose / mdl.period;
            int ndiv = data[np].ndiv;
            double norm = std::max(1, ndiv - 1);

            for (int nd = 0; nd < ndiv; ++nd) {
                double phi, wgt;
                if (ndiv == 1) { phi = phase; wgt = 1.0; }
                else {
                    phi = phase + expose * (nd - (ndiv - 1) / 2.0) / (ndiv - 1);
                    wgt = (nd == 0 || nd == ndiv - 1) ? 0.5 : 1.0;
                }

                Subs::Vec3 earth = Roche::set_earth(cosi, sini, phi);
                double phin = phi - floor(phi);
                double w1 = slfac * wgt * gint.scale1(phi) / norm;
                double w2 = slfac * wgt * gint.scale2(phi) / norm;

                // When coarse and fine grids alias, keep all phases in one
                // batch. Splitting them would traverse the very same grid
                // twice and launch a second OpenMP team for no benefit.
                PhaseBatch &b1 = copy1 ? pb1f
                                       : ((gint.type(phi) == 1) ? pb1f : pb1c);
                PhaseBatch &b2 = copy2 ? pb2f
                                       : ((gint.type(phi) == 3) ? pb2f : pb2c);
                b1.push(earth.x(), earth.y(), earth.z(), phin, w1, (int)np);
                b2.push(earth.x(), earth.y(), earth.z(), phin, w2, (int)np);
            }
        }

        // Sweep the grids (these parallelize internally over face blocks)
        vector<double> o1f(pb1f.size()), o1c(pb1c.size());
        vector<double> o2f(pb2f.size()), o2c(pb2c.size());
        if (pb1f.size())
            flat_sum_star1_multi(fg1f, pb1f, ldc1, mdl.beam_factor1,
                                 mdl.spin1, VFAC, XCOFM, o1f.data());
        if (pb1c.size())
            flat_sum_star1_multi(fg1c, pb1c, ldc1, mdl.beam_factor1,
                                 mdl.spin1, VFAC, XCOFM, o1c.data());
        if (pb2f.size())
            flat_sum_star2_multi(fg2f, pb2f, ldc2, mdl.beam_factor2,
                                 mdl.spin2, VFAC, XCOFM, mdl.glens1, rlens1,
                                 o2f.data());
        if (pb2c.size())
            flat_sum_star2_multi(fg2c, pb2c, ldc2, mdl.beam_factor2,
                                 mdl.spin2, VFAC, XCOFM, mdl.glens1, rlens1,
                                 o2c.data());

        // Scatter weighted sums back to the data points
        for (size_t np = 0; np < data.size(); np++) calc[np] = 0.;
        for (size_t k = 0; k < pb1f.size(); k++) calc[pb1f.idx[k]] += pb1f.w[k] * o1f[k];
        for (size_t k = 0; k < pb1c.size(); k++) calc[pb1c.idx[k]] += pb1c.w[k] * o1c[k];
        for (size_t k = 0; k < pb2f.size(); k++) calc[pb2f.idx[k]] += pb2f.w[k] * o2f[k];
        for (size_t k = 0; k < pb2c.size(); k++) calc[pb2c.idx[k]] += pb2c.w[k] * o2c[k];

        // Disc, edge and spot contributions (small grids) plus "third light"
        if (!disc.empty() || !edge.empty() || !spot.empty()) {
            #pragma omp parallel for schedule(dynamic,1)
            for (size_t np = 0; np < data.size(); np++) {
                double phase = phases[np], slfac = slfacs[np];
                double expose = data[np].expose / mdl.period;
                calc[np] += slfac * (comp_disc(mdl.iangle, mdl.lin_limb_disc,
                                               mdl.quad_limb_disc, phase, expose,
                                               data[np].ndiv, mdl.q, disc) +
                                     comp_edge(mdl.iangle, mdl.lin_limb_disc,
                                               mdl.quad_limb_disc, phase, expose,
                                               data[np].ndiv, mdl.q, edge) +
                                     comp_spot(mdl.iangle, phase, expose,
                                               data[np].ndiv, mdl.q, spot));
            }
        }
        for (size_t np = 0; np < data.size(); np++) calc[np] += mdl.third;
    }

    // Solvers only consume the fit and chi-square. Avoid another full face
    // sweep for the reporting-only white-dwarf contribution in their hot
    // iteration loops.
    wdwarf = diagnostics
        ? comp_star1_flat(mdl.iangle, ldc1, 0.5, 0., 1, mdl.q,
                          mdl.beam_factor1, mdl.velocity_scale,
                          gint, fg1f, fg1c)
        : 0.0;

    if (scale) {
        if (mdl.iscale) {
            vector<Subs::ddat> svd(data.size());
            vector<double> w;
            vector<vector<double>> u, v;
            for (size_t np = 0; np < data.size(); np++) {
                svd[np].x = data[np].time;
                svd[np].y = data[np].flux;
                if (data[np].weight <= 0.) {
                    svd[np].z = -1.;
                } else {
                    svd[np].z = data[np].ferr / sqrt(data[np].weight);
                }
            }

            // Compute scaling factors
            sfac.resize(mdl.t2 > 0 ? 5 : 4);

            Subs::svdfit(svd, sfac, fcomp, u, v, w);
            wdwarf *= sfac[0];
            if (mdl.t2 <= 0.) {
                vector<double> tfac(4);
                tfac = sfac;
                sfac.resize(5);
                sfac[0] = tfac[0];
                sfac[1] = tfac[1];
                sfac[2] = tfac[2];
                sfac[3] = tfac[3];
                sfac[4] = 0.;
            }

            // Calculate fit
            for (size_t np = 0; np < data.size(); np++) {
                calc[np] = sfac[0] * fcomp[np][0] + sfac[1] * fcomp[np][1] +
                           sfac[2] * fcomp[np][2] + sfac[3] * fcomp[np][3];
                if (mdl.t2 > 0.) calc[np] += sfac[4] * fcomp[np][4];
            }

            wnok = 0.;
            chisq = 0.;
            for (size_t i = 0; i < data.size(); i++) {
                if (data[i].weight > 0.) {
                    wnok += data[i].weight;
                    chisq += data[i].weight * Subs::sqr((data[i].flux -
                    calc[i]) / data[i].ferr);
                }
            }
        } else {
            double ssfac = re_scale(data, calc, chisq, wnok);
            wdwarf *= ssfac;
            sfac[0] = sfac[1] = sfac[2] = sfac[3] = sfac[4] = ssfac;
        }
    } else {
        wdwarf *= sfac[0];
        if (mdl.iscale) {
            for (size_t np = 0; np < data.size(); np++) {
                calc[np] = sfac[0] * fcomp[np][0] + sfac[1] * fcomp[np][1] +
                           sfac[2] * fcomp[np][2] + sfac[3] * fcomp[np][3];
                if (mdl.t2 > 0.) calc[np] += sfac[4] * fcomp[np][4];
            }
        } else {
            for (size_t np = 0; np < data.size(); np++)
                calc[np] *= sfac[0];
        }
        if (rdata) {
            // real data so compute chi**2
            wnok = 0.;
            chisq = 0.;
            for (size_t i = 0; i < data.size(); i++) {
                if (data[i].weight > 0.) {
                    wnok += data[i].weight;
                    chisq += data[i].weight * Subs::sqr((data[i].flux -
                    calc[i]) / data[i].ferr);
                }
            }
        }
    }
    
    // These four reductions are reporting diagnostics, not part of the fit.
    if (diagnostics) {
        logg1 = comp_gravity1(mdl, star1f);
        logg2 = comp_gravity2(mdl, star2f);
        rv1 = comp_radius1(star1f);
        rv2 = comp_radius2(star2f);
    } else {
        logg1 = logg2 = rv1 = rv2 = 0.0;
    }

    return;
}
