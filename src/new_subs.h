// Modern re-adaptations of ancient functions from trmrsh/cpp-subs

#ifndef NEW_SUBS_H
#define NEW_SUBS_H

#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <bit>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <array>
#include <cstdint>
#include <type_traits>
#include <fstream>
#include <random>
#pragma once
#include "lcurve_base/array1d.h"

using namespace std;


namespace Subs {
    template<typename T>
    T byte_swap(T value);

    class Subs_Error : public std::runtime_error {
    public:
        // Default constructor with empty message
        Subs_Error() : std::runtime_error("") {
        }

        // Constructor that stores a message
        Subs_Error(const std::string &str) : std::runtime_error(str) {
        }

        // No need to override what() unless you want custom behavior;
        // std::runtime_error::what() already works as expected
    };


    inline string toupper(const string &str) {
        string temp_str = str;
        transform(temp_str.begin(), temp_str.end(), temp_str.begin(), ::toupper);
        return temp_str;
    }

    inline bool string_to_bool(const string &entry) {
        string test = Subs::toupper(entry);
        if (test == "T" || test == "TRUE" || test == "Y" || test == "YES" || test == "1") {
            return true;
        }
        if (test == "F" || test == "FALSE" || test == "N" || test == "NO" || test == "0") {
            return false;
        }
        throw Subs_Error("to_stringing_to_bool: could not translate entry = " + entry + "to a bool");
    }

    template<class X>
    unsigned long int locate(const X *xx, unsigned long int n, X x) {
        // Ascending or descending check
        bool ascnd = (xx[n - 1] >= xx[0]);

        // Special cases: x matches the first or last element
        if (x == xx[0]) {
            return 1;
        } else if (x == xx[n - 1]) {
            return n - 1;
        }

        // Use lower_bound for ascending or upper_bound for descending order
        unsigned long int jhi;
        if (ascnd) {
            // Find the first element that is not less than x
            auto it = lower_bound(xx, xx + n, x);
            jhi = distance(xx, it);
        } else {
            // For descending order, we can use upper_bound to find the first element
            // greater than x, then adjust to get the correct position.
            auto it = upper_bound(xx, xx + n, x, greater<X>());
            jhi = distance(xx, it);
        }

        return jhi;
    }

    template <typename T>
    T select(std::vector<T>& arr, int k) {
        if (arr.empty()) throw std::invalid_argument("Array is empty");
        k = std::clamp(k, 0, static_cast<int>(arr.size()) - 1);
        std::nth_element(arr.begin(), arr.begin() + k, arr.end());
        return arr[k];
    }

    //! Return a with the same sign as b
    template <class X, class Y>
    inline X sign(const X& a, const Y& b){
        return(b >= 0. ? (X)fabs(a) : -(X)fabs(a));
    }

    template<class X>
    void hunt(const X *xx, unsigned long int n, X x, unsigned long int &jhi) {
        // Ascending or descending check
        bool ascnd = (xx[n - 1] > xx[0]);

        // Use lower_bound for ascending or upper_bound for descending order
        if (ascnd) {
            // Find the first element that is not less than x
            auto it = lower_bound(xx, xx + n, x);
            jhi = distance(xx, it);
        } else {
            // For descending order, we can use upper_bound to find the first element
            // greater than x, then adjust to get the correct position.
            auto it = upper_bound(xx, xx + n, x, greater<X>());
            jhi = distance(xx, it);
        }

        // Special case where x matches the first or last element
        if (x == xx[n - 1]) {
            jhi = n - 1;
        } else if (x == xx[0]) {
            jhi = 1;
        }
    }

    //! Abstract class for powell and amoeba
    /** This class is the base class for usage by the simplex minimisation routine amoeba and
     * similar routines that need to know a function value given an Array1D of parameters.
     */
    class Afunc {
    public:
        //! The function call
        /** This routine should return the function value after it has been passed
         * a vector of parameters. It is not defined as 'const' since you may well want to
         * alter the object during the computation.
         */
        virtual double operator()(const Array1D<double>& vec) = 0;
        virtual ~Afunc(){}
    };


    //! Abstract class for rtsafe function
    /** This class is the base class for usage by the 'rtsafe'. It declares the one
     * function that is needed.
     */
    class RTfunc {
    public:
        //! The function call
        /** This routine should compute the function value and derivative at position x.
         * \param x  the poisition to evaluate the function and derivative
         * \param f  the function value (returned)
         * \param fd the derivative (returned)
         */
        virtual void operator()(double x, double &f, double &fd) const = 0;

        virtual ~RTfunc() = default;
    };

    //! Abstract class for basic function object
    /** This class is the base class for usage by any routine reaquiring a basic function
     * object representing a 1D function. It declares one function that is needed which
     * returns the function value given the position. This is such a standard usage that it is
     * called Sfunc
     */
    class Sfunc {
    public:
        //! The function call
        /** This routine should return the function value at position x.
         * \param x  the position to evaluate the function and derivative
         * \return the function value
         */
        virtual double operator()(double x) = 0;

        virtual ~Sfunc() = default;
    };

    double gauss2(int seed);

    //! 1D minimisation routine without derivatives
    double brent(double xstart, double x1, double x2, Sfunc& f, double tol, double& xmin);

    //! 1D minimisation with derivatives
    double dbrent(double ax, double bx, double cx, Sfunc& func, Sfunc& dfunc, double acc,
                  bool stopfast, double pref, double& xmin);


    inline string str(const int &con, int ndig) {
        string result = to_string(con);
        int padding_needed = std::max(0, ndig - static_cast<int>(result.size()));

        if (padding_needed > 0) {
            result.insert(result.begin(), padding_needed, '0');
        }

        return result;
    }

    //! Convert degrees to radians
    inline double deg2rad(double deg) {
        return M_PI * deg / 180.;
    }

    //! Convert radians to degrees
    inline double rad2deg(double rad){
        return 180.*rad/M_PI;
    }

    class Vec3 {
    public:
        //! Default constructor (zero vector)
        Vec3() : xc(0.), yc(0.), zc(0.) {
        }

        //! General constructor from three numbers
        Vec3(double x, double y, double z) : xc(x), yc(y), zc(z) {
        }

        //! General constructor 3 element array
        Vec3(double *v) : xc(v[0]), yc(v[1]), zc(v[2]) {
        }

        //! Access to x
        double &x() { return xc; }

        //! Access to y
        double &y() { return yc; }

        //! Access to z
        double &z() { return zc; }

        //! Access to x
        const double &x() const { return xc; }

        //! Access to y
        const double &y() const { return yc; }

        //! Access to z
        const double &z() const { return zc; }

        //! Set to 3 numbers
        void set(double x, double y, double z) {
            xc = x;
            yc = y;
            zc = z;
        }

        //! Set to a 3 element array
        void set(double *v) {
            xc = v[0];
            yc = v[1];
            zc = v[2];
        }

        //! Get to a 3 element array
        void get(double *v) const {
            v[0] = xc;
            v[1] = yc;
            v[2] = zc;
        }

        //! Normalises a vector
        void unit() {
            double norm = xc * xc + yc * yc + zc * zc;
            if (norm > 0.) {
                norm = sqrt(norm);
                xc /= norm;
                yc /= norm;
                zc /= norm;
            } else {
                throw Subs_Error("void Subs::Vec3::unit(): null vector");
            }
        }

        //! Returns the length of a vector
        double length() const {
            return sqrt(xc * xc + yc * yc + zc * zc);
        }

        //! Returns the length squared of a vector
        double sqr() const {
            return xc * xc + yc * yc + zc * zc;
        }

        //! Sets a vector to be a unit vector in the X direction
        void unitx() {
            xc = 1.;
            yc = zc = 0.;
        }

        //! Sets a vector to be a unit vector in the Y direction
        void unity() {
            yc = 1.;
            xc = zc = 0.;
        }

        //! Sets a vector to be a unit vector in the Z direction
        void unitz() {
            zc = 1.;
            xc = yc = 0.;
        }

        //! Returns scalar or dot product of two vectors
        friend double dot(const Vec3 &vec1, const Vec3 &vec2) {
            return (vec1.xc * vec2.xc + vec1.yc * vec2.yc + vec1.zc * vec2.zc);
        }

        //! Returns cross product of two vectors
        friend Vec3 cross(const Vec3 &vec1, const Vec3 &vec2) {
            Vec3 temp;
            temp.xc = vec1.yc * vec2.zc - vec1.zc * vec2.yc;
            temp.yc = vec1.zc * vec2.xc - vec1.xc * vec2.zc;
            temp.zc = vec1.xc * vec2.yc - vec1.yc * vec2.xc;
            return temp;
        }

        //! Multiplication by a constant in place
        void operator*=(double con) {
            xc *= con;
            yc *= con;
            zc *= con;
        }

        //! Division by a constant in place
        void operator/=(double con) {
            xc /= con;
            yc /= con;
            zc /= con;
        }

        //! Addition of another Vec3 in place
        void operator+=(const Vec3 &vec) {
            xc += vec.xc;
            yc += vec.yc;
            zc += vec.zc;
        }

        //! Subtraction of another Vec3 in place
        void operator-=(const Vec3 &vec) {
            xc -= vec.xc;
            yc -= vec.yc;
            zc -= vec.zc;
        }

        //! Difference between two vectors
        friend Vec3 operator-(const Vec3 &vec1, const Vec3 &vec2) {
            Vec3 temp = vec1;
            temp -= vec2;
            return temp;
        }

        //! Sum of two vectors
        friend Vec3 operator+(const Vec3 &vec1, const Vec3 &vec2) {
            Vec3 temp = vec1;
            temp += vec2;
            return temp;
        }

        //! Multiplication by a constant
        friend Vec3 operator*(double con, const Vec3 &vec) {
            Vec3 temp = vec;
            temp *= con;
            return temp;
        }

        //! Division by a constant
        friend Vec3 operator/(const Vec3 &vec, double con) {
            Vec3 temp = vec;
            temp /= con;
            return temp;
        }

        //! ASCII output
        friend ostream &operator<<(ostream &ostr, const Vec3 &vec) {
            ostr << vec.xc << " " << vec.yc << " " << vec.zc;
            return ostr;
        }

        //! ASCII input
        friend istream &operator>>(istream &istr, Vec3 &vec) {
            istr >> vec.xc >> vec.yc >> vec.zc;
            return istr;
        }

        //! negative of a vector
        Vec3 operator-() {
            Vec3 temp;
            temp.xc = -xc;
            temp.yc = -yc;
            temp.zc = -zc;
            return temp;
        }

    private:
        double xc, yc, zc;
    };

    double dot(const Vec3 &vec1, const Vec3 &vec2);

    Vec3 cross(const Vec3 &vec1, const Vec3 &vec2);

    Vec3 operator-(const Vec3 &vec1, const Vec3 &vec2);

    Vec3 operator+(const Vec3 &vec1, const Vec3 &vec2);

    Vec3 operator*(double con, const Vec3 &vec);

    Vec3 operator/(const Vec3 &vec, double con);

    ostream &operator<<(ostream &ostr, const Vec3 &vec);

    istream &operator>>(istream &istr, Vec3 &vec);

    template<class X, class Y, class Z>
    struct xyz;

    template<class X, class Y, class Z>
    std::istream &operator>>(std::istream &ist, xyz<X, Y, Z> &obj);

    template<class X, class Y, class Z>
    std::ostream &operator<<(std::ostream &ost, const xyz<X, Y, Z> &obj);

    template <class X>
    inline X sqr(const X& a){
        return (a*a);
    }


    template <class X, class Y>
    struct xy;

    template <class X, class Y>
    std::istream& operator>>(std::istream& ist, xy<X,Y>& obj);

    template <class X, class Y>
    std::ostream& operator<<(std::ostream& ost, const xy<X,Y>& obj);

    //! General 2 variable structure

    /** This is completely general structure for storing pairs of numbers,
     * for instance time, velocity, error or whatever.
     */
    template <class X, class Y>
    struct xy{

        //! Default constructor
        xy() : x(0), y(0) {}

        //! Constructor from values
        xy(const X& xx, const Y& yy) : x(xx), y(yy) {}

        //! X value
        X x;

        //! Y value
        Y y;

        //! ASCII input
        friend std::istream& operator>><>(std::istream& ist, xy<X,Y>& obj);

        //! ASCII output
        friend std::ostream& operator<<<>(std::ostream& ost, const xy<X,Y>& obj);
    };

    /** ASCII input operator. Reads two numbers separated by spaces.
     * \param ist input stream
     * \param obj the 2 parameter object to read the data into
     * \return the input stream
     */
    template <class X, class Y>
    std::istream& operator>>(std::istream& ist, xy<X,Y>& obj){
        ist >> obj.x >> obj.y;
        return ist;
    }

    /** ASCII output operator. Writes two numbers separated by spaces.
     * \param ost output stream
     * \param obj the 2 parameter object to write out
     * \return the output stream
     */
    template <class X, class Y>
    std::ostream& operator<<(std::ostream& ost, const xy<X,Y>& obj){
        ost << obj.x << " " << obj.y;
        return ost;
    }

    //! General 3 variable structure

    /** This is completely general structure for storing triples of numbers,
     * for instance time, velocity, error or whatever.
     */
    template<class X, class Y, class Z>
    struct xyz {
        //! Default constructor
        xyz() : x(0), y(0), z(0) {
        }

        //! Constructor from values
        xyz(const X &xx, const Y &yy, const Z &zz) : x(xx), y(yy), z(zz) {
        }

        //! X value
        X x;

        //! Y value

        Y y;

        //! Z value
        Z z;

        //! ASCII input
        friend std::istream &operator>><>(std::istream &ist, xyz<X, Y, Z> &obj);

        //! ASCII output
        friend std::ostream &operator<<<>(std::ostream &ost, const xyz<X, Y, Z> &obj);
    };

    /** ASCII input operator. Reads three numbers separated by spaces.
     * \param ist input stream
     * \param obj the 3 parameter object to read the data into
     * \return the input stream
     */
    template<class X, class Y, class Z>
    std::istream &operator>>(std::istream &ist, xyz<X, Y, Z> &obj) {
        ist >> obj.x >> obj.y >> obj.z;
        return ist;
    }

    /** ASCII output operator. Writes three numbers separated by spaces.
     * \param ost output stream
     * \param obj the 3 parameter object to write out
     * \return the output stream
     */
    template<class X, class Y, class Z>
    std::ostream &operator<<(std::ostream &ost, const xyz<X, Y, Z> &obj) {
        ost << obj.x << " " << obj.y << " " << obj.z;
        return ost;
    }

    //! Defines a particular instance of an xyz<X,Y,Z> object suited to radial velocity work
    typedef xyz<double, float, float> rv;

    //! Defines a particular instance of an xyz<X,Y,Z> object suited for more accurate work
    typedef xyz<double, double, float> ddat;

    class Format {
    public:
        class Bound_form_d;
        class Bound_form_f;
        class Bound_form_s;

        //! Constructor
        explicit Format(int precision = 8);

        //! Create and object with a format bound to a double
        Bound_form_d operator()(double d) const;

        //! Create and object with a format bound to a float
        Bound_form_f operator()(float f) const;

        //! Create and object with a format bound to a string
        Bound_form_s operator()(const string &s) const;

        //! Create and object with a format bound to a string, changing width on the fly
        Bound_form_s operator()(const string &s, int width);

        //! Sets the precision
        void precision(int p);

        //! Sets scientific notation
        void scientific();

        //! Sets fixed format
        void fixed();

        //! Sets best variable output format
        void general();

        //! Sets minimum field width
        void width(int w);

        //! Print decimal point
        void showpoint();

        //! Do not print decimal point unless necessary
        void noshowpoint();

        //! Makes anything uppercase it can
        void uppercase();

        //! Makes anything lowercase it can
        void lowercase();

        //! Sets the fill character
        void fill(char c);

        //! Pads after value
        void left();

        //! Pads before value
        void right();

        //! Pads between number and its sign
        void internal();

        //! Output of double
        friend ostream &operator<<(ostream &ostr, const Bound_form_d &bf);

        //! Output of float
        friend ostream &operator<<(ostream &ostr, const Bound_form_f &bf);

        //! Output of string
        friend ostream &operator<<(ostream &ostr, const Bound_form_s &bf);

    private:
        int precision_;
        int width_;
        ios::fmtflags format_;
        bool upper;
        bool showpnt;
        char fill_char;
        ios::fmtflags fadjust;
    };

    //! Burlisch-Stoer routine
    bool bsstep(double y[], double dydx[], int nv, double& xx,
                double htry, double eps, double yscal[], double &hdid,
                double &hnext,
                void (*derivs)(double, double [], double []));

    //! Modified mid-point routine
    void  mmid(double y[], double dydx[], int nvar, double xs,
               double htot, int nstep, double yout[],
               void (*derivs)(double, double[], double[]));


    //! Abstract class for bsstep function object
    class Bsfunc {

    public:

        //! The function call
        /** Evaluates derivatives dydt given current time t and coordinates y
         * use function object to store other parameters needed
         */
        virtual void operator()(double t, double y[], double dydt[]) const = 0;
        virtual ~Bsfunc(){}
    };

    bool bsstep(double y[], double dydx[], int nv, double &xx,
                double htry, double eps, double yscal[],
                double &hdid, double &hnext, const Bsfunc& derivs);

    //! Modified mid-point routine
    void  mmid(double y[], double dydx[], int nvar, double xs,
               double htot, int nstep, double yout[],
               const Bsfunc& derivs);

    // Alternative for mmid for conservative 2nd order equations
    bool bsstepst(double y[], double dydx[], int nv, double &xx,
                  double htry, double eps, double yscal[],
                  double &hdid, double &hnext, const Bsfunc& derivs);

    void stoerm(double y[], double d2y[], int nvar, double xs, double htot,
                int nstep, double yout[], const Bsfunc& derivs);

    //! Polynomial extrapolation routine
    void  pzextr(int iest, double xest, double yest[], double yz[],
                 double dy[], int nv);


    //! Combination of format and a double
    struct Bound_form_d {
        //! the format
        const Format &form;
        //! the double value
        double val;
        //! Constructor
        Bound_form_d(const Format &format, double value) : form(format), val(value) {
        }
    };

    struct Bound_form_f {
        //! the format
        const Format &form;
        //! the double value
        float val;
        //! Constructor
        Bound_form_f(const Format &format, float value) : form(format), val(value) {
        }
    };

    struct Bound_form_s {
        //! the format
        const Format &form;
        //! the string
        string val;
        //! Constructor
        Bound_form_s(const Format &format, const string &value) : form(format), val(value) {
        }
    };

    ostream &operator<<(ostream &ostr, const Bound_form_d &bf);

    ostream &operator<<(ostream &ostr, const Bound_form_f &bf);

    ostream &operator<<(ostream &ostr, const Bound_form_s &bf);


    //! Singular value decomposition fitting
    double svdfit(const vector<rv> &data, vector<float> &a,
                  const vector<vector<float>> &vect, vector<vector<float>> &u,
                  vector<vector<float>> &v, vector<float> &w);

    //! Singular value decomposition fitting
    double svdfit(const vector<ddat> &data, vector<double> &a, const vector<vector<double>> &vect,
                  vector<vector<double>> &u, vector<vector<double>> &v, vector<double> &w);

    template<class T>
    T sign(T a, T b) {
        return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);
    }

    template<class T>
    T pythag(T a, T b) {
        T absa = std::abs(a);
        T absb = std::abs(b);
        if (absa > absb) {
            T temp = absb / absa;
            return absa * std::sqrt(1.0 + temp * temp);
        } else {
            return (absb == 0.0 ? 0.0 : absb * std::sqrt(1.0 + (absa / absb) * (absa / absb)));
        }
    }

    //! Singular value decomposition
    /**
     * svdcmp performs singular value decomposition.
     * Given an M x N matrix A this routine computes its
     * singular value decomposition A = U.W.V^t. The matrix U
     * replaces A on output. The diagonal matrix W is returned
     * as a vector and the matrix V is returned in the last argument (not
     * the transpose).
     * \param a M x N matrix A. Declare e.g. a(M,N). M might be the number of data point and N the
     * number of polynomials for example. Returns with elements of the matrix U.
     * \param w N element vector of diagonal elements of centre matrix W
     * \param v N by N elements of matrix V
     */
    template<class T>
    void svdcmp(std::vector<std::vector<T>>& a, std::vector<T>& w, std::vector<std::vector<T>>& v) {
        int m = static_cast<int>(a.size());
        int n = static_cast<int>(a[0].size());
        
        w.resize(n);
        v.assign(n, std::vector<T>(n, 0.0));
        
        int flag, i, its, j, jj, k, l = 0, nm = 0;
        T anorm, c, f, g, h, s, scale, x, y, z;

        std::vector<T> rv1(n); // Work space array
        g = scale = anorm = T(0);

        for (i = 0; i < n; i++) {
            l = i + 1;
            rv1[i] = scale * g;
            g = s = scale = T(0);
            if (i < m) {
                for (k = i; k < m; k++) scale += std::abs(a[k][i]);
                if (scale != T(0)) {
                    for (k = i; k < m; k++) {
                        a[k][i] /= scale;
                        s += a[k][i] * a[k][i];
                    }
                    f = a[i][i];
                    g = -sign(std::sqrt(s), f);
                    h = f * g - s;
                    a[i][i] = f - g;
                    for (j = l; j < n; j++) {
                        for (s = T(0), k = i; k < m; k++) s += a[k][i] * a[k][j];
                        f = s / h;
                        for (k = i; k < m; k++) a[k][j] += f * a[k][i];
                    }
                    for (k = i; k < m; k++) a[k][i] *= scale;
                }
            }
            w[i] = scale * g;
            g = s = scale = T(0);
            if (i < m && i != n - 1) {
                for (k = l; k < n; k++) scale += std::abs(a[i][k]);
                if (scale != T(0)) {
                    for (k = l; k < n; k++) {
                        a[i][k] /= scale;
                        s += a[i][k] * a[i][k];
                    }
                    f = a[i][l];
                    g = -sign(std::sqrt(s), f);
                    h = f * g - s;
                    a[i][l] = f - g;
                    for (k = l; k < n; k++) rv1[k] = a[i][k] / h;
                    for (j = l; j < m; j++) {
                        for (s = T(0), k = l; k < n; k++) s += a[j][k] * a[i][k];
                        for (k = l; k < n; k++) a[j][k] += s * rv1[k];
                    }
                    for (k = l; k < n; k++) a[i][k] *= scale;
                }
            }
            anorm = std::max(anorm, T(std::abs(w[i]) + std::abs(rv1[i])));
        }
        
        for (i = n - 1; i >= 0; i--) {
            if (i < n - 1) {
                if (g != T(0)) {
                    for (j = l; j < n; j++)
                        v[j][i] = (a[i][j] / a[i][l]) / g;
                    for (j = l; j < n; j++) {
                        for (s = T(0), k = l; k < n; k++) s += a[i][k] * v[k][j];
                        for (k = l; k < n; k++) v[k][j] += s * v[k][i];
                    }
                }
                for (j = l; j < n; j++) v[i][j] = v[j][i] = T(0);
            }
            v[i][i] = T(1);
            g = rv1[i];
            l = i;
        }
        
        for (i = std::min(m, n) - 1; i >= 0; i--) {
            l = i + 1;
            g = w[i];
            for (j = l; j < n; j++) a[i][j] = T(0);
            if (g != T(0)) {
                g = T(1) / g;
                for (j = l; j < n; j++) {
                    for (s = T(0), k = l; k < m; k++) s += a[k][i] * a[k][j];
                    f = (s / a[i][i]) * g;
                    for (k = i; k < m; k++) a[k][j] += f * a[k][i];
                }
                for (j = i; j < m; j++) a[j][i] *= g;
            } else {
                for (j = i; j < m; j++) a[j][i] = T(0);
            }
            ++a[i][i];
        }
        
        for (k = n - 1; k >= 0; k--) {
            for (its = 1; its <= 30; its++) {
                flag = 1;
                for (l = k; l >= 0; l--) {
                    nm = l - 1;
                    if (T(std::abs(rv1[l]) + anorm) == anorm) {
                        flag = 0;
                        break;
                    }
                    if (T(std::abs(w[nm]) + anorm) == anorm) break;
                }
                if (flag) {
                    c = T(0);
                    s = T(1);
                    for (i = l; i <= k; i++) {
                        f = s * rv1[i];
                        rv1[i] *= c;
                        if (T(std::abs(f) + anorm) == anorm) break;
                        g = w[i];
                        h = pythag(f, g);
                        w[i] = h;
                        h = T(1) / h;
                        c = g * h;
                        s = -f * h;
                        for (j = 0; j < m; j++) {
                            y = a[j][nm];
                            z = a[j][i];
                            a[j][nm] = y * c + z * s;
                            a[j][i] = z * c - y * s;
                        }
                    }
                }
                z = w[k];
                if (l == k) {
                    if (z < T(0)) {
                        w[k] = -z;
                        for (j = 0; j < n; j++) v[j][k] = -v[j][k];
                    }
                    break;
                }
                if (its == 30)
                    std::cerr << "No convergence in svdcmp in 30 iterations\n";
                x = w[l];
                nm = k - 1;
                y = w[nm];
                g = rv1[nm];
                h = rv1[k];
                f = ((y - z) * (y + z) + (g - h) * (g + h)) / (T(2) * h * y);
                g = pythag(f, T(1));
                f = ((x - z) * (x + z) + h * ((y / (f + sign(g, f))) - h)) / x;
                c = s = T(1);
                for (j = l; j <= nm; j++) {
                    i = j + 1;
                    g = rv1[i];
                    y = w[i];
                    h = s * g;
                    g = c * g;
                    z = pythag(f, h);
                    rv1[j] = z;
                    c = f / z;
                    s = h / z;
                    f = x * c + g * s;
                    g = g * c - x * s;
                    h = y * s;
                    y *= c;
                    for (jj = 0; jj < n; jj++) {
                        x = v[jj][j];
                        z = v[jj][i];
                        v[jj][j] = x * c + z * s;
                        v[jj][i] = z * c - x * s;
                    }
                    z = pythag(f, h);
                    w[j] = z;
                    if (z != T(0)) {
                        z = T(1) / z;
                        c = f * z;
                        s = h * z;
                    }
                    f = c * g + s * y;
                    x = c * y - s * g;
                    for (jj = 0; jj < m; jj++) {
                        y = a[jj][j];
                        z = a[jj][i];
                        a[jj][j] = y * c + z * s;
                        a[jj][i] = z * c - y * s;
                    }
                }
                rv1[l] = T(0);
                rv1[k] = f;
                w[k] = x;
            }
        }
    }

    //! Singular value decomposition, back substitution
    /** Solves u x = b for vectors x (N elements) and b (M elements),
     *  where M x N matrix u has been transformed into its singular value
     *  decomposition by svdcmp
     *
     * \param u M x N matrix from svdcmp. Declare e.g. u(M,N). M might be the number of data point and N the number of polynomials for example.
     * \param w N element vector from svdcmp. Should have been edited to set small values = 0
     * \param v N x N matrix from svdcmp
     * \param b M element target vector
     * \param x  element solution vector
     */
    template<class T>
    void svbksb(const std::vector<std::vector<T>>& u, const std::vector<T>& w, 
                const std::vector<std::vector<T>>& v, const std::vector<T>& b,
                std::vector<T>& x) {
        size_t jj, j, i, m = u.size(), n = u[0].size();
        T s;
        x.resize(n);
        std::vector<T> tmp(n); // Work space array
        
        for (j = 0; j < n; j++) {
            s = T(0);
            if (w[j] != T(0)) {
                for (i = 0; i < m; i++) s += u[i][j] * b[i];
                s /= w[j];
            }
            tmp[j] = s;
        }
        
        for (j = 0; j < n; j++) {
            s = T(0);
            for (jj = 0; jj < n; jj++) s += v[j][jj] * tmp[jj];
            x[j] = s;
        }
    }


    //! Singular value decomposition fitting
    double svdfit(const std::vector<rv> &data, vector<float> &a,
                  const std::vector<std::vector<float>> &vect, std::vector<std::vector<float>> &u,
                  std::vector<std::vector<float>> &v, vector<float> &w);

    //! Singular value decomposition fitting
    double svdfit(const std::vector<ddat> &data, vector<double> &a, const vector<vector<double>> &vect,
                  vector<vector<double>> &u, vector<vector<double>> &v, vector<double> &w);

    //! Singular value decomposition fitting
    double svdfit(const std::vector<rv> &data, vector<float> &a,
                  const vector<double> &cosine, const vector<double> &sine,
                  std::vector<std::vector<float>> &u, std::vector<std::vector<float>> &v, vector<float> &w);

    //! Planck function Bnu.
    double planck(double wave, double temp);

    //! Logarithmic derivative of Planck function Bnu wrt wavelength
    double dplanck(double wave, double temp);

    //! Logarithmic derivative of Planck function Bnu wrt T
    double dlpdlt(double wave, double temp);

    //! Simplex minimisation routine
    void amoeba(std::vector<std::pair<Array1D<double>, double> >& params, double ftol, int nmax, Afunc& func, int& nfunc);
};

#endif //NEW_SUBS_H
