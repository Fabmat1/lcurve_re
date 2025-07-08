

#ifndef MASS_RATIO_PDF_H
#define MASS_RATIO_PDF_H

class MassRatioPDFGrid;
extern MassRatioPDFGrid* g_mass_ratio_grid;

void initialize_mass_ratio_pdf_grid(
    double m1_mean, double m1_err,
    double m2_mean, double m2_err,
    double K_mean, double K_err,
    double R_mean, double R_err,
    double P_mean, double P_err,
    double incl_min = 15.0, double incl_max = 90.0, int n_incl = 200,
    int n_q = 500, int n_vs = 500, int n_rs = 500, int nsamp = 30000
);

double log_mass_ratio_pdf(double inclination, double q, double v_s, double r_s);

// Individual PDF access functions
double mass_ratio_pdf(double inclination, double q);
double velocity_scale_pdf(double inclination, double v_s);
double radius_scale_pdf(double inclination, double r_s);

// Utility function for mass ratio calculation
double mass_ratio_from_inclination(double inclination, double mass1, double min_mass2);

// Cleanup
void cleanup_mass_ratio_pdf_grid();

// Additional utility functions for testing
void print_grid_info();
bool check_in_bounds(double inclination, double q, double v_s, double r_s);

#endif