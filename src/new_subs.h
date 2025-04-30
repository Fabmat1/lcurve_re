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
#include "lcurve_base/array1d.h"
#include "lcurve_base/buffer2d.h"
#include <array>
#include <cstdint>
#include <type_traits>

using namespace std;


namespace Subs {
    template<typename T>
        requires (std::is_integral_v<T> || std::is_floating_point_v<T>)
    constexpr T byte_swap(T value) noexcept {
        if constexpr (std::is_integral_v<T>) {
            // Directly use the library byte-swap for integers
            return byteswap(value);
        } else {
            // For floats/doubles: cast to unsigned integer of same size,
            // byteswap that, then cast back
            using U = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
            U as_int = std::bit_cast<U>(value);
            as_int = byteswap(as_int);
            return std::bit_cast<T>(as_int);
        }
    }

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
        throw Subs_Error("Subs::string_to_bool: could not translate entry = " + entry + "to a bool");
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

    template<typename T>
    T select(vector<T> &arr, int k) {
        k = max(0, min(static_cast<int>(arr.size()) - 1, k));

        nth_element(arr.begin(), arr.begin() + k, arr.end());
        return arr[k];
    }

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

    inline string str(const int &con, int ndig) {
        string result = to_string(con);
        int padding_needed = std::max(0, ndig - static_cast<int>(result.size()));

        if (padding_needed > 0) {
            result.insert(result.begin(), padding_needed, '0');
        }

        return result;
    }

    inline double deg2rad(double deg) {
        return M_PI * deg / 180.;
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
    double svdfit(const Buffer1D<rv> &data, Buffer1D<float> &a,
                  const Buffer2D<float> &vect, Buffer2D<float> &u,
                  Buffer2D<float> &v, Buffer1D<float> &w);

    //! Singular value decomposition fitting
    double svdfit(const Buffer1D<ddat> &data, Buffer1D<double> &a, const Buffer2D<double> &vect,
                  Buffer2D<double> &u, Buffer2D<double> &v, Buffer1D<double> &w);

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
    template<class X>
    void svdcmp(Buffer2D<X> &a, Buffer1D<X> &w, Buffer2D<X> &v) {
        int m = a.nrow(), n = a.ncol();
        w.resize(n);
        v.resize(n, n);
        int flag, i, its, j, jj, k, l = 0, nm = 0;
        X anorm, c, f, g, h, s, scale, x, y, z;

        Buffer1D<X> rv1(n); // Work space array
        g = scale = anorm = 0.;

        for (i = 0; i < n; i++) {
            l = i + 1;
            rv1[i] = scale * g;
            g = s = scale = 0.;
            if (i < m) {
                for (k = i; k < m; k++) scale += fabs(a[k][i]);
                if (scale) {
                    for (k = i; k < m; k++) {
                        a[k][i] /= scale;
                        s += a[k][i] * a[k][i];
                    }
                    f = a[i][i];
                    g = -sign(sqrt(s), f);
                    h = f * g - s;
                    a[i][i] = f - g;
                    for (j = l; j < n; j++) {
                        for (s = 0., k = i; k < m; k++) s += a[k][i] * a[k][j];
                        f = s / h;
                        for (k = i; k < m; k++) a[k][j] += f * a[k][i];
                    }
                    for (k = i; k < m; k++) a[k][i] *= scale;
                }
            }
            w[i] = scale * g;
            g = s = scale = 0.;
            if (i < m && i != n - 1) {
                for (k = l; k < n; k++) scale += fabs(a[i][k]);
                if (scale) {
                    for (k = l; k < n; k++) {
                        a[i][k] /= scale;
                        s += a[i][k] * a[i][k];
                    }
                    f = a[i][l];
                    g = -sign(sqrt(s), f);
                    h = f * g - s;
                    a[i][l] = f - g;
                    for (k = l; k < n; k++) rv1[k] = a[i][k] / h;
                    for (j = l; j < m; j++) {
                        for (s = 0., k = l; k < n; k++) s += a[j][k] * a[i][k];
                        for (k = l; k < n; k++) a[j][k] += s * rv1[k];
                    }
                    for (k = l; k < n; k++) a[i][k] *= scale;
                }
            }
            anorm = std::max(anorm, X(fabs(w[i]) + fabs(rv1[i])));
        }
        for (i = n - 1; i >= 0; i--) {
            if (i < n - 1) {
                if (g) {
                    for (j = l; j < n; j++)
                        v[j][i] = (a[i][j] / a[i][l]) / g;
                    for (j = l; j < n; j++) {
                        for (s = 0., k = l; k < n; k++) s += a[i][k] * v[k][j];
                        for (k = l; k < n; k++) v[k][j] += s * v[k][i];
                    }
                }
                for (j = l; j < n; j++) v[i][j] = v[j][i] = 0.;
            }
            v[i][i] = 1.;
            g = rv1[i];
            l = i;
        }
        for (i = std::min(m, n) - 1; i >= 0; i--) {
            l = i + 1;
            g = w[i];
            for (j = l; j < n; j++) a[i][j] = 0.;
            if (g) {
                g = 1. / g;
                for (j = l; j < n; j++) {
                    for (s = 0., k = l; k < m; k++) s += a[k][i] * a[k][j];
                    f = (s / a[i][i]) * g;
                    for (k = i; k < m; k++) a[k][j] += f * a[k][i];
                }
                for (j = i; j < m; j++) a[j][i] *= g;
            } else for (j = i; j < m; j++) a[j][i] = 0.;
            ++a[i][i];
        }
        for (k = n - 1; k >= 0; k--) {
            for (its = 1; its <= 30; its++) {
                flag = 1;
                for (l = k; l >= 0; l--) {
                    nm = l - 1;
                    if (X(fabs(rv1[l]) + anorm) == anorm) {
                        flag = 0;
                        break;
                    }
                    if (X(fabs(w[nm]) + anorm) == anorm) break;
                }
                if (flag) {
                    c = 0.;
                    s = 1.;
                    for (i = l; i <= k; i++) {
                        f = s * rv1[i];
                        rv1[i] *= c;
                        if (X(fabs(f) + anorm) == anorm) break;
                        g = w[i];
                        h = pythag(f, g);
                        w[i] = h;
                        h = 1. / h;
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
                    if (z < 0.) {
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
                f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2. * h * y);
                g = pythag(f, X(1));
                f = ((x - z) * (x + z) + h * ((y / (f + sign(g, f))) - h)) / x;
                c = s = 1.;
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
                    if (z) {
                        z = 1. / z;
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
                rv1[l] = 0.;
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
    template<class X>
    void svbksb(const Buffer2D<X> &u, const Buffer1D<X> &w, const Buffer2D<X> &v, const Buffer1D<X> &b,
                Buffer1D<X> &x) {
        size_t jj, j, i, m = u.nrow(), n = u.ncol();
        X s;

        x.resize(n);
        Buffer1D<X> tmp(n); // Work space array

        for (j = 0; j < n; j++) {
            s = 0.0;
            if (w[j]) {
                for (i = 0; i < m; i++) s += u[i][j] * b[i];
                s /= w[j];
            }
            tmp[j] = s;
        }

        for (j = 0; j < n; j++) {
            s = 0.0;
            for (jj = 0; jj < n; jj++) s += v[j][jj] * tmp[jj];
            x[j] = s;
        }
    }

    //! Singular value decomposition fitting
    double svdfit(const Buffer1D<rv> &data, Buffer1D<float> &a,
                  const Buffer2D<float> &vect, Buffer2D<float> &u,
                  Buffer2D<float> &v, Buffer1D<float> &w);

    //! Singular value decomposition fitting
    double svdfit(const Buffer1D<ddat> &data, Buffer1D<double> &a, const Buffer2D<double> &vect,
                  Buffer2D<double> &u, Buffer2D<double> &v, Buffer1D<double> &w);

    //! Singular value decomposition fitting
    double svdfit(const Buffer1D<rv> &data, Buffer1D<float> &a,
                  const Buffer1D<double> &cosine, const Buffer1D<double> &sine,
                  Buffer2D<float> &u, Buffer2D<float> &v, Buffer1D<float> &w);
};

#endif //NEW_SUBS_H
