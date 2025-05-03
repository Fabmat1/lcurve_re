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

using namespace std;


namespace Subs {
    template <class T> class Array1D;
    template <class T> class Array2D;
    template <class T> class Buffer1D;
    template <class T> class Buffer2D;

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

    //! Buffer class for handling memory.
    /** A class designed to supply a safe 1D array, 'safe' in
     * that it is deallocated whenever the object goes out of scope.
     * It creates a pointer to an array which can then be used
     * in the usual way as an array. The pointer is automatically
     * deleted when its Buffer1D goes out of scope. It also stores the number
     * of pixels and the number of memory elements allocated. The latter can
     * be larger than the number of pixels to make extension of the array size
     * more efficient. Buffer1D is designed to handle any type of data and therefore
     * does not supply operations such as addition; look at Array1D for such
     * specialisation which inherits Buffer1D and adds such facilities. Buffer1D can
     * therefore contain complex objects as its elements. Such objects will need to
     * support the operations of assignment and ASCII I/O. Binary I/O functions
     * write, read and skip are provided but should only be used on objects with
     * no pointers because they will store the pointers but not whatever they
     * point to.
     */
    template <class X>
    class Buffer1D {
    public:

        //! Default constructor, makes NULL pointer.
        Buffer1D() : buff(NULL), npix(0), nmem(0) {}

        //! Constructor grabbing space for npix points
        Buffer1D(int npix);

        //! Constructor grabbing space for npix points but with nmem elements allocated
        Buffer1D(int npix, int nmem);

        //! Constructor from a file name
        Buffer1D(const std::string& file);

        //! Copy constructor
        Buffer1D(const Buffer1D& obj);

        //! Constructor from a vector
        Buffer1D(const std::vector<X>& obj);

        //! Destructor
        virtual ~Buffer1D(){
            if(buff != NULL)
                delete[] buff;
        }

        //! Assignment
        Buffer1D& operator=(const Buffer1D& obj);

        //! Assignment
        Buffer1D& operator=(const X& con);

        //! Returns the number of pixels
        int get_npix() const {return npix;}

        //! Returns the number of pixels
        int size() const {return npix;}

        //! Returns the number of pixels allocated in memory
        int mem() const {return nmem;}

        //! Change number of pixels
        void resize(int npix);

        //! Zeroes number of elements in the array (but does not change memory allocation)
        void clear(){npix = 0;}

        //! Change number of pixels directly, no other effect (experts only)
        void set_npix(int npix){
            if(npix > nmem)
                throw runtime_error("Subs::Buffer1D::set_npix(int): attempt to set npix > nmem not allowed.");
            this->npix = npix;
        }

        //! Element access
        const X& operator[](int i) const {
            return buff[i];
        }

        //! Element access
        X& operator[](int i) {
            return buff[i];
        }

        //! Add another value to the end of the buffer, preserving all other data
        void push_back(const X& value);

        //! Remove a pixel
        void remove(int index);

        //! Insert a pixel
        void insert(int index, const X& value);

        //! Conversion operator for interfacing with normal C style arrays.
        operator X*(){return buff;}

        //! Returns pointer to the buffer to interface with functions requiring normal C style arrays.
        X* ptr() {return buff;}

        //! Returns pointer to the buffer to interface with functions requiring normal C style arrays.
        const X* ptr() const {return buff;}

        //! Read from an ASCII file
        void load_ascii(const std::string& file);

        //! Write out a Buffer1D to a binary file
        void write(std::ostream& s) const;

        //! Skip a Buffer1D in a binary file
        static void skip(std::istream& s, bool swap_bytes);

        //! Read a poly from a binary file
        void read(std::istream& s, bool swap_bytes);

        //friend std::ostream& operator<<<>(std::ostream& s, const Buffer1D& vec);

        //friend std::istream& operator>><>(std::istream& s, Buffer1D& vec);

    protected:

        //! The pointer; used extensively in Array1D, change at your peril!
        X* buff;

        //! For derived class ASCII input
        virtual void ascii_input(std::istream& s);

        //! For derived class ASCII output
        virtual void ascii_output(std::ostream& s) const;

    private:

        // number of pixels and number of memory elements
        int npix;
        int nmem;

    };


    /** This constructor gets space for exactly npix points
     * \param npix the number of points to allocate space for
     */
    template <class X>
    Buffer1D<X>::Buffer1D(int npix) : npix(npix), nmem(npix) {
        if(npix < 0)
            throw runtime_error("Subs::Buffer1D<>(int): attempt to allocate < 0 point");
        if(npix == 0){
            buff = NULL;
        }else{
            if((buff = new(std::nothrow) X [nmem]) == NULL){
                this->npix = nmem = 0;
                throw runtime_error("Subs::Buffer1D::Buffer1D(int): failed to allocate memory");
            }
        }
    }

    /** This constructor gets space for exactly npix points
     * \param npix the number of pixels
     * \param nmem the number of memory elements
     */
    template <class X>
    Buffer1D<X>::Buffer1D(int npix, int nmem) : npix(npix), nmem(nmem) {
        if(npix < 0)
            throw runtime_error("Subs::Buffer1D<>(int, int): attempt to set < 0 pixels");
        if(nmem < npix)
            throw runtime_error("Subs::Buffer1D<>(int, int): must allocate at least as many memory elements as pixels");
        if(nmem == 0){
            buff = NULL;
        }else{
            if((buff = new(std::nothrow) X [nmem]) == NULL){
                this->npix = nmem = 0;
                throw runtime_error("Subs::Buffer1D<>(int, int): failure to allocate " + to_string(nmem) + " points.");
            }
        }
    }

    /** Constructor by reading a file
     * \param file the file to read, an ASCII file.
     */
    template <class X>
    Buffer1D<X>::Buffer1D(const std::string& file) : buff(NULL), npix(0), nmem(0) {
        try{
            load_ascii(file);
        }
        catch(const runtime_error& err){
            throw runtime_error("Buffer1D<X>::Buffer1D(const std::string&): error constructing from a file " + string(err.what()));
        }
    }

    /** Copy constructor to make an element by element copy of an object
     */
    template <class X>
    Buffer1D<X>::Buffer1D(const Buffer1D<X>& obj) : npix(obj.npix), nmem(obj.npix) {
        if(nmem == 0){
            buff = NULL;
        }else{
            if((buff = new(std::nothrow) X [nmem]) == NULL){
                npix = nmem = 0;
                throw runtime_error("Subs::Buffer1D<>(const Buffer1D<>&): failure to allocate " + to_string(nmem) + " points.");
            }
            for(int i=0; i<npix; i++)
                buff[i] = obj.buff[i];
        }
    }

    /** Constructor to make an element by element copy of a vector
     */
    template <class X>
    Buffer1D<X>::Buffer1D(const std::vector<X>& obj) : npix(obj.size()), nmem(obj.size()) {
        if(nmem == 0){
            buff = NULL;
        }else{
            if((buff = new(std::nothrow) X [nmem]) == NULL){
                npix = nmem = 0;
                throw runtime_error("Subs::Buffer1D<>(const std::vector<>&): failure to allocate " + to_string(nmem) + " points.");
            }
            for(int i=0; i<npix; i++)
                buff[i] = obj[i];
        }
    }

    /** Sets one Buffer1D equal to another.
     */
    template <class X>
    Buffer1D<X>& Buffer1D<X>::operator=(const Buffer1D<X>& obj){

        if(this == &obj) return *this;

        // First check whether we can avoid reallocation of memory
        if(buff != NULL){
            if(obj.npix <= nmem){
                npix = obj.npix;
                for(int i=0; i<npix; i++)
                    buff[i] = obj.buff[i];
                return *this;
            }else{
                delete[] buff;
            }
        }

        // Allocate memory
        npix = nmem = obj.npix;
        if(nmem == 0){
            buff = NULL;
        }else{
            if((buff = new(std::nothrow) X [nmem]) == NULL){
                npix = nmem = 0;
                throw runtime_error("Subs::Buffer1D<>(const Buffer1D<>&): failure to allocate " + to_string(nmem) + " points.");
            }
        }

        // Finally copy
        for(int i=0; i<npix; i++)
            buff[i] = obj.buff[i];

        return *this;
    }

    /** Sets a Buffer1D to a constant
     */
    template <class X>
    Buffer1D<X>& Buffer1D<X>::operator=(const X& con){

        for(int i=0; i<npix; i++)
            buff[i] = con;

        return *this;
    }

    /** This changes the number of pixels. It does not preserve the data in general.
     * \param npix the new array size
     */
    template <class X>
    void Buffer1D<X>::resize(int npix){
        if(buff != NULL){
            if(npix <= nmem){
                this->npix = npix;
                return;
            }else{
                delete[] buff;
            }
        }

        this->npix = nmem = npix;
        if(nmem < 0)
            throw runtime_error("Subs::Buffer1D::resize(int): attempt to allocate < 0 points");
        if(nmem == 0){
            buff = NULL;
            return;
        }
        if((buff = new(std::nothrow) X [nmem]) == NULL){
            this->npix = nmem = 0;
            throw runtime_error("Subs::Buffer1D::resize(int): failed to allocate new memory");
        }
    }

    /** This routine adds a new value to the end of a buffer, increasing the 
     * memory allocated if need be.
     * \param value new value to add to the end
     */
    template <class X>
    void Buffer1D<X>::push_back(const X& value){
        if(npix < nmem){
            buff[npix] = value;
            npix++;
        }else{
            nmem *= 2;
            nmem  = (nmem == 0) ? 1 : nmem;
            X* temp;
            if((temp = new(std::nothrow) X [nmem]) == NULL){
                nmem /= 2;
                throw runtime_error("Subs::Buffer1D::push_back(const X&): failed to extend memory");
            }
            for(int i=0; i<npix; i++)
                temp[i] = buff[i];
            temp[npix] = value;
            npix++;
            delete[] buff;
            buff = temp;
        }
    }

    /** This routine removes a pixel at a given index. The memory allocated
     * is not changed.
     * \param index the pixel to be removed
     */
    template <class X>
    void Buffer1D<X>::remove(int index){
        npix--;
        for(int i=index; i<npix; i++)
            buff[i] = buff[i+1];
    }

    /** This routine inserts a pixel at a given index. The memory allocated
     * may have to increase
     * \param index the pixel to be removed
     */
    template <class X>
    void Buffer1D<X>::insert(int index, const X& value){

        if(npix < nmem){
            for(int i=npix; i>index; i--)
                buff[i] = buff[i-1];
            buff[index] = value;
            npix++;
        }else{
            nmem *= 2;
            nmem  = (nmem == 0) ? 1 : nmem;
            X* temp;
            if((temp = new(std::nothrow) X [nmem]) == NULL){
                nmem /= 2;
                throw runtime_error("Subs::Buffer1D::insert(int, const X&): failed to extend memory");
            }
            for(int i=0; i<index; i++)
                temp[i] = buff[i];
            for(int i=npix; i>index; i--)
                temp[i] = buff[i-1];
            temp[index] = value;
            npix++;
            delete[] buff;
            buff = temp;
        }
    }

    //! Binary output
    template <class X>
    void Buffer1D<X>::write(std::ostream& s) const {
        s.write((char*)&npix, sizeof(int));
        s.write((char*)buff,  sizeof(X[npix]));
    }

    //! Binary input
    template <class X>
    void Buffer1D<X>::read(std::istream& s, bool swap_bytes) {
        s.read((char*)&npix, sizeof(int));
        if(!s) return;
        if(swap_bytes) npix = Subs::byte_swap(npix);
        this->resize(npix);
        s.read((char*)buff,  sizeof(X[npix]));
        if(swap_bytes) Subs::byte_swap(buff, npix);
    }

    //! Binary skip
    template <class X>
    void Buffer1D<X>::skip(std::istream& s, bool swap_bytes) {
        int npixel;
        s.read((char*)&npixel, sizeof(int));
        if(!s) return;
        if(swap_bytes) npixel = Subs::byte_swap(npixel);
        s.ignore(sizeof(X[npixel]));
    }

    /* Loads data into a Buffer1D from an ASCII file with one
     * element per line. The elements must support ASCII input.
     * Define a suitable structure for complex input. Lines starting with
     * # are skipped.
     * \param file the file name to load
     */
    template <class Type>
    void Buffer1D<Type>::load_ascii(const std::string& file){

        std::ifstream fin(file.c_str());
        if(!fin)
            throw runtime_error("void Buffer1D<>::load~_ascii(const std::string&): could not open " + file);

        // Clear the buffer
        this->resize(0);
        Type line;
        char c;
        while(fin){
            c = fin.peek();
            if(!fin) break;
            if(c == '#' || c == '\n'){
                while(fin.get(c)) if(c == '\n') break;
            }else{
                if(fin >> line) this->push_back(line);
                while(fin.get(c)) if(c == '\n') break; // ignore the rest of the line
            }
        }
        fin.close();

    }

    template <class X>
    std::ostream& operator<<(std::ostream& s, const Buffer1D<X>& vec){
        vec.ascii_output(s);
        return s;
    }

    //! Singular value decomposition fitting
    double svdfit(const Buffer1D<rv> &data, Buffer1D<float> &a,
                  const Buffer2D<float> &vect, Buffer2D<float> &u,
                  Buffer2D<float> &v, Buffer1D<float> &w);

    //! Singular value decomposition fitting
    double svdfit(const Buffer1D<ddat> &data, Buffer1D<double> &a, const Buffer2D<double> &vect,
                  Buffer2D<double> &u, Buffer2D<double> &v, Buffer1D<double> &w);

    //! Compute sqrt(a*a+b*b) avoiding under/over flows
    template <class X> X pythag(const X& a, const X& b){
        X absa, absb, temp;
        absa = fabs(a);
        absb = fabs(b);
        if(absa > absb){
            temp = absb / absa;
            return absa*sqrt(1.+temp*temp);
        }else if(absb == 0.){
            return 0.;
        }else{
            temp = absa / absb;
            return absb*sqrt(1.+temp*temp);
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

    //! Planck function Bnu.
    double planck(double wave, double temp);

    //! Logarithmic derivative of Planck function Bnu wrt wavelength
    double dplanck(double wave, double temp);

    //! Logarithmic derivative of Planck function Bnu wrt T
    double dlpdlt(double wave, double temp);

    template<class X>
    void Buffer1D<X>::ascii_output(std::ostream &s) const {
        if (!s) return;
        s << this->size();
        for (int i = 0; i < this->size(); i++)
            s << " " << (*this)[i];
    }

    template<class X>
    std::istream &operator>>(std::istream &s, Buffer1D<X> &vec) {
        vec.ascii_input(s);
        return s;
    }

    template<class X>
    void Buffer1D<X>::ascii_input(std::istream &s) {
        if (!s) return;
        int nelem;
        s >> nelem;
        this->resize(nelem);
        for (int i = 0; i < this->size(); i++)
            s >> (*this)[i];
    }

};

#endif //NEW_SUBS_H
