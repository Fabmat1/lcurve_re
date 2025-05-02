#include <cstdlib>
#include <cmath>
#include <string>
#include "../new_subs.h"
#include "roche.h"


double Roche::xl11(double q, double spin){
    constexpr int NMAX   = 1000;
    constexpr double EPS = 1.e-12;

    double ssq = spin*spin;
    double x, xold, f, df, mu;
    double a1, a2, a3, a4, a5, a6, d1, d2, d3, d4, d5;
    int n;

    if(q <= 0.)
        throw Roche_Error("Roche::xl11(double, double): q = " + to_string(q) + " <= 0.");

    // Set poly coefficients
    mu = q/(1.+q);
    a1 = -1.+mu;
    d1 = 1.*(a2 = 2.-2.*mu);
    d2 = 2.*(a3 = -1.+mu);
    d3 = 3.*(a4 = ssq+2.*mu);
    d4 = 4.*(a5 = -2.*ssq-mu);
    d5 = 5.*(a6 = ssq);

    n = 0;
    xold = 0.;
    x    = 1./(1.+ q);
    while(n < NMAX && fabs(x-xold) > EPS*fabs(x)){
        xold = x;
        f = x*(x*(x*(x*(x*a6+a5)+a4)+a3)+a2)+a1;
        df = x*(x*(x*(x*d5+d4)+d3)+d2)+d1;
        x -= f/df;
        n++;
    }
    if(n == NMAX)
        throw Roche_Error("Roche::xl11(double, double): exceeded maximum iterations");
    return x;
}


double Roche::xl12(double q, double spin){

    const int NMAX   = 1000;
    const double EPS = 1.e-12;

    double ssq = spin*spin;
    double x, xold, f, df, mu;
    double a1, a2, a3, a4, a5, a6, d1, d2, d3, d4, d5;
    int n;

    if(q <= 0.)
        throw Roche_Error("Roche::xl12(double, double): q = " + to_string(q) + " <= 0.");

    // Set poly coefficients
    mu = q/(1.+q);
    a1 = -1.+mu;
    d1 = 1.*(a2 =  2.-2.*mu);
    d2 = 2.*(a3 = -ssq+mu);
    d3 = 3.*(a4 =  3.*ssq+2.*mu-2.);
    d4 = 4.*(a5 = 1.-mu-3.*ssq);
    d5 = 5.*(a6 =  ssq);

    n = 0;
    xold = 0.;
    x    = 1./(1.+ q);
    while(n < NMAX && fabs(x-xold) > EPS*fabs(x)){
        xold = x;
        f    = x*(x*(x*(x*(x*a6+a5)+a4)+a3)+a2)+a1;
        df   = x*(x*(x*(x*d5+d4)+d3)+d2)+d1;
        x   -= f/df;
        n++;
    }
    if(n == NMAX)
        throw Roche_Error("Roche::xl12(double,double): exceeded maximum iterations");
    return x;
}

