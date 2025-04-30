#ifndef ARRAY1D_H
#define ARRAY1D_H

#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include "../new_subs.h"

using namespace std;


namespace Subs {
    //! Template 1D array class

    /**
     * Subs::Array1D<X> is a template 1D numerical array class based upon
     * the memory handling object Buffer1D. This class should be used if you want
     * to add, subtract, multiply etc vectors (and the data type X can support it).
     * This typically means that you can use Array1D for X=float, double etc but not
     * more complex types. For the latter Buffer1D is the usual choice.
     */
    // Base error class

    // Derived error class
    class Buffer1D_Error : public runtime_error {
    public:
        // Default constructor
        Buffer1D_Error() : runtime_error("") {
        }

        // Constructor which stores a message
        Buffer1D_Error(const string &str) : runtime_error(str) {
        }
    };

    template<class X>
    class Buffer1D {
    public:
        //! Default constructor, makes NULL pointer.
        Buffer1D() : buff(NULL), npix(0), nmem(0) {
        }

        //! Constructor grabbing space for npix points
        Buffer1D(int npix);

        //! Constructor grabbing space for npix points but with nmem elements allocated
        Buffer1D(int npix, int nmem);

        //! Constructor from a file name
        Buffer1D(const std::string &file);

        //! Copy constructor
        Buffer1D(const Buffer1D &obj);

        //! Constructor from a vector
        Buffer1D(const std::vector<X> &obj);

        //! Destructor
        virtual ~Buffer1D() {
            if (buff != NULL)
                delete[] buff;
        }

        //! Assignment
        Buffer1D &operator=(const Buffer1D &obj);

        //! Assignment
        Buffer1D &operator=(const X &con);

        //! Returns the number of pixels
        int get_npix() const { return npix; }

        //! Returns the number of pixels
        int size() const { return npix; }

        //! Returns the number of pixels allocated in memory
        int mem() const { return nmem; }

        //! Change number of pixels
        void resize(int npix);

        //! Zeroes number of elements in the array (but does not change memory allocation)
        void clear() { npix = 0; }

        //! Change number of pixels directly, no other effect (experts only)
        void set_npix(int npix) {
            if (npix > nmem)
                throw Buffer1D_Error("Subs::Buffer1D::set_npix(int): attempt to set npix > nmem not allowed.");
            this->npix = npix;
        }

        //! Element access
        const X &operator[](int i) const {
            return buff[i];
        }

        //! Element access
        X &operator[](int i) {
            return buff[i];
        }

        //! Add another value to the end of the buffer, preserving all other data
        void push_back(const X &value);

        //! Remove a pixel
        void remove(int index);

        //! Insert a pixel
        void insert(int index, const X &value);

        //! Conversion operator for interfacing with normal C style arrays.
        operator X *() { return buff; }

        //! Returns pointer to the buffer to interface with functions requiring normal C style arrays.
        X *ptr() { return buff; }

        //! Returns pointer to the buffer to interface with functions requiring normal C style arrays.
        const X *ptr() const { return buff; }

        //! Read from an ASCII file
        void load_ascii(const std::string &file);

        //! Write out a Buffer1D to a binary file
        void write(std::ostream &s) const;

        //! Skip a Buffer1D in a binary file
        static void skip(std::istream &s, bool swap_bytes);

        //! Read a poly from a binary file
        void read(istream &s, bool swap_bytes);

        //friend ostream &operator<<<>(ostream &s, const Buffer1D &vec);

        //friend istream &operator>><>(istream &s, Buffer1D &vec);

    protected:
        //! The pointer; used extensively in Array1D, change at your peril!
        X *buff;

        //! For derived class ASCII input
        virtual void ascii_input(std::istream &s);

        //! For derived class ASCII output
        virtual void ascii_output(std::ostream &s) const;

    private:
        // number of pixels and number of memory elements
        int npix;
        int nmem;
    };


    /** This constructor gets space for exactly npix points
     * \param npix the number of points to allocate space for
     */
    template<class X>
    Buffer1D<X>::Buffer1D(int npix) : npix(npix), nmem(npix) {
        if (npix < 0)
            throw Buffer1D_Error("Subs::Buffer1D<>(int): attempt to allocate < 0 point");
        if (npix == 0) {
            buff = NULL;
        } else {
            if ((buff = new(std::nothrow) X [nmem]) == NULL) {
                this->npix = nmem = 0;
                throw Buffer1D_Error("Subs::Buffer1D::Buffer1D(int): failed to allocate memory");
            }
        }
    }

    /** This constructor gets space for exactly npix points
     * \param npix the number of pixels
     * \param nmem the number of memory elements
     */
    template<class X>
    Buffer1D<X>::Buffer1D(int npix, int nmem) : npix(npix), nmem(nmem) {
        if (npix < 0)
            throw Buffer1D_Error("Subs::Buffer1D<>(int, int): attempt to set < 0 pixels");
        if (nmem < npix)
            throw Buffer1D_Error(
                "Subs::Buffer1D<>(int, int): must allocate at least as many memory elements as pixels");
        if (nmem == 0) {
            buff = NULL;
        } else {
            if ((buff = new(std::nothrow) X [nmem]) == NULL) {
                this->npix = nmem = 0;
                throw Buffer1D_Error("Subs::Buffer1D<>(int, int): failure to allocate " + to_string(nmem) + " points.");
            }
        }
    }

    /** Constructor by reading a file
     * \param file the file to read, an ASCII file.
     */
    template<class X>
    Buffer1D<X>::Buffer1D(const std::string &file) : buff(NULL), npix(0), nmem(0) {
        try {
            load_ascii(file);
        } catch (const Buffer1D_Error &err) {
            throw Buffer1D_Error("Buffer1D<X>::Buffer1D(const std::string&): error constructing from a file ");
        }
    }

    /** Copy constructor to make an element by element copy of an object
     */
    template<class X>
    Buffer1D<X>::Buffer1D(const Buffer1D<X> &obj) : npix(obj.npix), nmem(obj.npix) {
        if (nmem == 0) {
            buff = NULL;
        } else {
            if ((buff = new(std::nothrow) X [nmem]) == NULL) {
                npix = nmem = 0;
                throw Buffer1D_Error(
                    "Subs::Buffer1D<>(const Buffer1D<>&): failure to allocate " + to_string(nmem) + " points.");
            }
            for (int i = 0; i < npix; i++)
                buff[i] = obj.buff[i];
        }
    }

    /** Constructor to make an element by element copy of a vector
     */
    template<class X>
    Buffer1D<X>::Buffer1D(const std::vector<X> &obj) : npix(obj.size()), nmem(obj.size()) {
        if (nmem == 0) {
            buff = NULL;
        } else {
            if ((buff = new(std::nothrow) X [nmem]) == NULL) {
                npix = nmem = 0;
                throw Buffer1D_Error(
                    "Subs::Buffer1D<>(const std::vector<>&): failure to allocate " + to_string(nmem) + " points.");
            }
            for (int i = 0; i < npix; i++)
                buff[i] = obj[i];
        }
    }

    /** Sets one Buffer1D equal to another.
     */
    template<class X>
    Buffer1D<X> &Buffer1D<X>::operator=(const Buffer1D<X> &obj) {
        if (this == &obj) return *this;

        // First check whether we can avoid reallocation of memory
        if (buff != NULL) {
            if (obj.npix <= nmem) {
                npix = obj.npix;
                for (int i = 0; i < npix; i++)
                    buff[i] = obj.buff[i];
                return *this;
            } else {
                delete[] buff;
            }
        }

        // Allocate memory
        npix = nmem = obj.npix;
        if (nmem == 0) {
            buff = NULL;
        } else {
            if ((buff = new(std::nothrow) X [nmem]) == NULL) {
                npix = nmem = 0;
                throw Buffer1D_Error(
                    "Subs::Buffer1D<>(const Buffer1D<>&): failure to allocate " + to_string(nmem) + " points.");
            }
        }

        // Finally copy
        for (int i = 0; i < npix; i++)
            buff[i] = obj.buff[i];

        return *this;
    }

    /** Sets a Buffer1D to a constant
     */
    template<class X>
    Buffer1D<X> &Buffer1D<X>::operator=(const X &con) {
        for (int i = 0; i < npix; i++)
            buff[i] = con;

        return *this;
    }

    /** This changes the number of pixels. It does not preserve the data in general.
     * \param npix the new array size
     */
    template<class X>
    void Buffer1D<X>::resize(int npix) {
        if (buff != NULL) {
            if (npix <= nmem) {
                this->npix = npix;
                return;
            } else {
                delete[] buff;
            }
        }

        this->npix = nmem = npix;
        if (nmem < 0)
            throw Buffer1D_Error("Subs::Buffer1D::resize(int): attempt to allocate < 0 points");
        if (nmem == 0) {
            buff = NULL;
            return;
        }
        if ((buff = new(std::nothrow) X [nmem]) == NULL) {
            this->npix = nmem = 0;
            throw Buffer1D_Error("Subs::Buffer1D::resize(int): failed to allocate new memory");
        }
    }

    /** This routine adds a new value to the end of a buffer, increasing the
     * memory allocated if need be.
     * \param value new value to add to the end
     */
    template<class X>
    void Buffer1D<X>::push_back(const X &value) {
        if (npix < nmem) {
            buff[npix] = value;
            npix++;
        } else {
            nmem *= 2;
            nmem = (nmem == 0) ? 1 : nmem;
            X *temp;
            if ((temp = new(std::nothrow) X [nmem]) == NULL) {
                nmem /= 2;
                throw Buffer1D_Error("Subs::Buffer1D::push_back(const X&): failed to extend memory");
            }
            for (int i = 0; i < npix; i++)
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
    template<class X>
    void Buffer1D<X>::remove(int index) {
        npix--;
        for (int i = index; i < npix; i++)
            buff[i] = buff[i + 1];
    }

    /** This routine inserts a pixel at a given index. The memory allocated
     * may have to increase
     * \param index the pixel to be removed
     */
    template<class X>
    void Buffer1D<X>::insert(int index, const X &value) {
        if (npix < nmem) {
            for (int i = npix; i > index; i--)
                buff[i] = buff[i - 1];
            buff[index] = value;
            npix++;
        } else {
            nmem *= 2;
            nmem = (nmem == 0) ? 1 : nmem;
            X *temp;
            if ((temp = new(std::nothrow) X [nmem]) == NULL) {
                nmem /= 2;
                throw Buffer1D_Error("Subs::Buffer1D::insert(int, const X&): failed to extend memory");
            }
            for (int i = 0; i < index; i++)
                temp[i] = buff[i];
            for (int i = npix; i > index; i--)
                temp[i] = buff[i - 1];
            temp[index] = value;
            npix++;
            delete[] buff;
            buff = temp;
        }
    }

    //! Binary output
    template<class X>
    void Buffer1D<X>::write(std::ostream &s) const {
        s.write((char *) &npix, sizeof(int));
        s.write((char *) buff, sizeof(X[npix]));
    }

    //! Binary input
    template<class X>
    void Buffer1D<X>::read(std::istream &s, bool swap_bytes) {
        s.read((char *) &npix, sizeof(int));
        if (!s) return;
        if (swap_bytes) npix = Subs::byte_swap(npix);
        this->resize(npix);
        s.read((char *) buff, sizeof(X[npix]));
        if (swap_bytes) Subs::byte_swap(buff, npix);
    }

    //! Binary skip
    template<class X>
    void Buffer1D<X>::skip(std::istream &s, bool swap_bytes) {
        int npixel;
        s.read((char *) &npixel, sizeof(int));
        if (!s) return;
        if (swap_bytes) npixel = Subs::byte_swap(npixel);
        s.ignore(sizeof(X[npixel]));
    }

    /* Loads data into a Buffer1D from an ASCII file with one
     * element per line. The elements must support ASCII input.
     * Define a suitable structure for complex input. Lines starting with
     * # are skipped.
     * \param file the file name to load
     */
    template<class Type>
    void Buffer1D<Type>::load_ascii(const std::string &file) {
        ifstream fin(file.c_str());
        if (!fin.is_open())
            throw Buffer1D_Error("void Buffer1D<>::load~_ascii(const std::string&): could not open " + file);

        // Clear the buffer
        this->resize(0);
        Type line;
        char c;
        while (fin) {
            c = fin.peek();
            if (!fin) break;
            if (c == '#' || c == '\n') {
                while (fin.get(c)) if (c == '\n') break;
            } else {
                if (fin >> line) this->push_back(line);
                while (fin.get(c)) if (c == '\n') break; // ignore the rest of the line
            }
        }
        fin.close();
    }

    template<class X>
    std::ostream &operator<<(std::ostream &s, const Buffer1D<X> &vec) {
        vec.ascii_output(s);
        return s;
    }

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


    template<class X>
    class Array1D : public Buffer1D<X> {
    public:
        //! Default constructor
        Array1D() : Buffer1D<X>() {
        }

        //! Constructor of an npix element vector
        Array1D(int npix) : Buffer1D<X>(npix) {
        }

        //! Constructs an npix element vector but with memory up to nmem
        Array1D(int npix, int nmem) : Buffer1D<X>(npix, nmem) {
        }

        //! Copy constructor
        Array1D(const Array1D &obj) : Buffer1D<X>(obj) {
        }

        //! Constructs an Array from a vector
        template<class Y>
        Array1D(const std::vector<Y> &vec);

        //! Constructor from an ordinary array
        template<class Y>
        Array1D(int npix, const Y *vec);

        //! Constructor from a function of the array index
        template<class Y>
        Array1D(int npix, const Y &func);

        //! Constructor from a file
        Array1D(const std::string &file) : Buffer1D<X>(file) {
        };

        //! Assign to a constant
        Array1D<X> &operator=(const X &con);

        //! Addition of a constant, in place
        void operator+=(const X &con);

        //! Subtraction of a constant, in place
        void operator-=(const X &con);

        //! Division by a constant, in place
        void operator/=(const X &con);

        //! Multiplication by a constant, in place
        void operator*=(const X &con);

        //! Addition of another array, in place
        template<class Y>
        void operator+=(const Array1D<Y> &vec);

        //! Subtraction of another array, in place
        template<class Y>
        void operator-=(const Array1D<Y> &vec);

        //! Multiplication by another array, in place
        template<class Y>
        void operator*=(const Array1D<Y> &vec);

        //! Division by another array, in place
        template<class Y>
        void operator/=(const Array1D<Y> &vec);

        //! Returns maximum value
        X max() const;

        //! Returns minimum value
        X min() const;

        //! Takes cosine of array
        void cos();

        //! Takes sine of array
        void sin();

        //! Determines whether values are monotonic
        bool monotonic() const;

        //! Locate a value in an ordered array
        void hunt(const X &x, int &jhi) const;

        //! Locate a value in an ordered array
        unsigned long locate(const X &x) const;

        //! Sorts an array into ascending order and returns a key to the original order
        Buffer1D<unsigned long int> sort();

        //! Return percentile (pcent from 0 to 100)
        X centile(double pcent);

        //! Return value of k-th smallest element (scrambles element order!)
        X select(int k);

        //! Returns median (scrambles element order!)
        X median();

        //! Returns sum
        X sum() const;

        //! Returns mean
        X mean() const;

        //! Returns length in Euclidean sense
        X length() const;
    };

    //! Error class
    class Array1D_Error : public Subs_Error {
    public:
        Array1D_Error() : Subs_Error("") {
        };

        Array1D_Error(const std::string &str) : Subs_Error(str) {
        };
    };

    /** Constructor of an Array1D from the STL vector class. It must be
     * possible to assign the two types to each other.
     * \param vec the vector to construct from. The Array1D will have the same number
     * of elements with the same values as the vector.
     */
    template<class X>
    template<class Y>
    Array1D<X>::Array1D(const std::vector<Y> &vec) : Buffer1D<X>(vec.size()) {
        for (int i = 0; i < this->size(); i++)
            this->buff[i] = vec[i];
    }

    /** Constructor of an Array1D from a standard C-like array. It must be
     * possible to assign the two types to each other.
     * \param vec the vector to construct from. The Array1D will have the same number
     * of elements with the same values as the vector.
     */
    template<class X>
    template<class Y>
    Array1D<X>::Array1D(int npix, const Y *vec) : Buffer1D<X>(npix) {
        for (int i = 0; i < this->size(); i++)
            this->buff[i] = vec[i];
    }

    /** Constructor of an Array1D from a function of the pixel index, with the first
     * pixel = 0.
     * \param func the function to construct from. The function must support a call of the
     * form func(int i) which returns the value at pixel i. Here is an example of a function (as
     * a function object class) that adds a simple offset
     * class Func {
     * public:
     * Func(int off) : offset(off) {}
     * int operator()(int i) const {return i+offset;}
     * private:
     * int offset;
     *};
     */
    template<class X>
    template<class Y>
    Array1D<X>::Array1D(int npix, const Y &func) : Buffer1D<X>(npix) {
        for (int i = 0; i < this->size(); i++)
            this->buff[i] = func(i);
    }

    // Operations with constants
    template<class X>
    void Array1D<X>::operator+=(const X &con) {
        for (int i = 0; i < this->size(); i++)
            this->buff[i] += con;
    }

    template<class X>
    void Array1D<X>::operator-=(const X &con) {
        for (int i = 0; i < this->size(); i++)
            this->buff[i] -= con;
    }

    template<class X>
    void Array1D<X>::operator*=(const X &con) {
        for (int i = 0; i < this->size(); i++)
            this->buff[i] *= con;
    }

    template<class X>
    void Array1D<X>::operator/=(const X &con) {
        for (int i = 0; i < this->size(); i++)
            this->buff[i] /= con;
    }

    // Returns maximum value
    template<class X>
    X Array1D<X>::max() const {
        if (this->size() == 0)
            throw Array1D_Error("X Array1D<X>::max(): null array, cannot take maximum");

        X vmax = this->buff[0];
        for (int i = 1; i < this->size(); i++)
            vmax = vmax < this->buff[i] ? this->buff[i] : vmax;
        return vmax;
    }

    // Returns minimum value
    template<class X>
    X Array1D<X>::min() const {
        if (this->size() == 0)
            throw Array1D_Error("X Array1D<X>::min(): null array, cannot take minimum");

        X vmin = this->buff[0];
        for (int i = 1; i < this->size(); i++)
            vmin = vmin > this->buff[i] ? this->buff[i] : vmin;
        return vmin;
    }

    /** This function returns the value of the k-th smallest element of an Array1D. It uses
     * a routine that scrambles the order for speed. Copy the Array1D first if the
     * order is important to you. The operation can be done by sorting, but this is faster
     * if you only want a single value.
     */
    template<class X>
    X Array1D<X>::select(int k) {
        return Subs::select(this->buff, this->size(), k);
    }

    /** This function returns the value of a percentile of an Array1D. It uses
     * a routine that scrambles the order for speed. Copy the Array1D first if the
     * order is important to you. The operation can be done by sorting, but this is faster
     * if you only want a single value.
     */
    template<class X>
    X Array1D<X>::centile(double pcent) {
        int k = int(pcent / 100 * this->size());
        k = k < 0 ? 0 : k;
        k = k < this->size() ? k : this->size() - 1;
        return Subs::select(this->buff, this->size(), k);
    }

    /** This function computes the median. It uses 'select' which scrambles the order.
     * Copy the Array1D first if the order is important to you. The median is clearly defined
     * for odd numbers of elements; for even numbers this routine averages the two middle values
     * (thus requiring two calls to 'select'and therefore slower than a similar odd number case.
     */
    template<class X>
    X Array1D<X>::median() {
        if (this->size() % 2 == 0) {
            return (Subs::select(this->buff, this->size(), this->size() / 2 - 1) +
                    Subs::select(this->buff, this->size(), this->size() / 2)) / 2;
        } else {
            return Subs::select(this->buff, this->size(), this->size() / 2);
        }
    }

    /** This function computes the mean. It returns zero if there are no elements
     *
     */
    template<class X>
    X Array1D<X>::mean() const {
        X sum = 0;
        if (this->size()) {
            for (int i = 0; i < this->size(); i++)
                sum += this->buff[i];
            sum /= this->size();
        }
        return sum;
    }

    /** This function computes the sum. It returns zero if there are no elements
     *
     */
    template<class X>
    X Array1D<X>::sum() const {
        X sum = 0;
        if (this->size()) {
            for (int i = 0; i < this->size(); i++)
                sum += this->buff[i];
        }
        return sum;
    }

    /** This function computes the length as the square root of the sum of squares
     */
    template<class X>
    X Array1D<X>::length() const {
        X sum = 0;
        for (int i = 0; i < this->size(); i++)
            sum += this->buff[i] * this->buff[i];
        return sqrt(sum);
    }

    // Operations with other arrays
    template<class X>
    template<class Y>
    void Array1D<X>::operator+=(const Array1D<Y> &vec) {
        if (this->size() != vec.size())
            throw Array1D_Error(
                "void Array1D<X>::operator+=(const Array1D<Y>& vec): incompatible numbers of elements, " + Subs::str(
                    this->size()) +
                " versus " + Subs::str(vec.size()));
        for (int i = 0; i < this->size(); i++)
            this->buff[i] += vec.buff[i];
    }

    template<class X>
    template<class Y>
    void Array1D<X>::operator-=(const Array1D<Y> &vec) {
        if (this->size() != vec.size())
            throw Array1D_Error(
                "void Array1D<X>::operator-=(const Array1D<Y>& vec): incompatible numbers of elements, " + Subs::str(
                    this->size()) +
                " versus " + Subs::str(vec.size()));
        for (int i = 0; i < this->size(); i++)
            this->buff[i] -= vec.buff[i];
    }

    template<class X>
    template<class Y>
    void Array1D<X>::operator*=(const Array1D<Y> &vec) {
        if (this->size() != vec.size())
            throw Array1D_Error(
                "void Array1D<X>::operator*=(const Array1D<Y>& vec): incompatible numbers of elements, " + Subs::str(
                    this->size()) +
                " versus " + Subs::str(vec.size()));
        for (int i = 0; i < this->size(); i++)
            this->buff[i] *= vec.buff[i];
    }

    template<class X>
    template<class Y>
    void Array1D<X>::operator/=(const Array1D<Y> &vec) {
        if (this->size() != vec.size())
            throw Array1D_Error(
                "void Array1D<X>::operator/=(const Array1D<Y>& vec): incompatible numbers of elements, " + Subs::str(
                    this->size()) +
                " versus " + Subs::str(vec.size()));
        for (int i = 0; i < this->size(); i++)
            this->buff[i] /= vec.buff[i];
    }

    template<class X>
    void Array1D<X>::cos() {
        for (int i = 0; i < this->size(); i++)
            this->buff[i] = std::cos(this->buff[i]);
    }

    template<class X>
    void Array1D<X>::sin() {
        for (int i = 0; i < this->size(); i++)
            this->buff[i] = std::sin(this->buff[i]);
    }

    /** Finds position of x assuming values are ordered. If they are not ordered, the routine will not
     * check, and wrong results will ensue. This one is useful if you have some idea of the likely
     * index value.
     * \param x the value to locate
     * \param jhi input roughly the index expected; returned as the value such that x lies between the jhi-1 and jhi'th
     * elements
     */
    template<class X>
    void Array1D<X>::hunt(const X &x, int &jhi) const {
        unsigned long int uljhi = jhi;
        Subs::hunt(this->buff, this->size(), x, uljhi);
        jhi = int(uljhi);
    }

    /** Finds position of x assuming values are ordered. If they are not ordered, the routine will not
     * check, and wrong results will ensue. This one is useful if you no idea the likely index value.
     * \param x the value to locate
     * \param jhi returned as the value such that x lies between the jhi-1 and jhi'th elements
     */
    template<class X>
    unsigned long Array1D<X>::locate(const X &x) const {
        return Subs::locate(this->buff, this->size(), x);
    }

    /** Sorts an array into ascending order
     */
    template<class X>
    Buffer1D<unsigned long int> Array1D<X>::sort() {
        Buffer1D<unsigned long int> key(this->size());
        heaprank(this->buff, key.ptr(), this->size());
        Array1D<X> temp(*this);
        for (int i = 0; i < this->size(); i++)
            this->buff[i] = temp[key[i]];
        return key;
    }

    /** Tests whether values increase or decrease monotonically and so whether hunt can be used.
     */
    template<class X>
    bool Array1D<X>::monotonic() const {
        if (this->size() < 3) {
            return true;
        } else {
            bool up = this->buff[0] < this->buff[this->size() - 1];
            for (int i = 1; i < this->size(); i++) {
                bool up_now = this->buff[i - 1] < this->buff[i];
                if ((up && !up_now) || (!up && up_now)) return false;
            }
            return true;
        }
    }

    // Non-member functions

    // Operations with other arrays
    template<class X, class Y>
    Array1D<X> operator-(const Array1D<X> &v1, const Array1D<Y> &v2) {
        if (v1.size() != v2.size())
            throw Array1D_Error(
                "void operator-=(const Array1D<X>&, const Array1D<Y>&): incompatible numbers of elements, " + Subs::str(
                    v1.size())
                + " versus " + Subs::str(v2.size()));
        Array1D<X> temp = v1;
        temp -= v2;
        return temp;
    }

    // Operations with other arrays
    template<class X, class Y>
    Array1D<X> operator+(const Array1D<X> &v1, const Array1D<Y> &v2) {
        if (v1.size() != v2.size())
            throw Array1D_Error(
                "void operator+=(const Array1D<X>&, const Array1D<Y>&): incompatible numbers of elements, " + Subs::str(
                    v1.size())
                + " versus " + Subs::str(v2.size()));
        Array1D<X> temp = v1;
        temp += v2;
        return temp;
    }

    //! Returns the maximum value
    template<class X>
    X max(const Array1D<X> &vec) {
        return vec.max();
    }

    //! Returns the minimum value
    template<class X>
    X min(const Array1D<X> &vec) {
        return vec.min();
    }

    //! Subtracts a constant to create a new Array1D
    template<class X, class Y>
    Array1D<X> operator-(const Array1D<X> &vec, const Y &con) {
        Array1D<X> temp = vec;
        temp -= con;
        return temp;
    }

    //! Pre-multiplies by a constant
    template<class X, class Y>
    Array1D<X> operator*(const Y &con, const Array1D<X> &vec) {
        Array1D<X> temp = vec;
        temp *= con;
        return temp;
    }

    //! Divides by a constant
    template<class X, class Y>
    Array1D<X> operator/(const Array1D<X> &vec, const Y &con) {
        Array1D<X> temp = vec;
        temp /= con;
        return temp;
    }

    //! Mulitplies two Array1Ds, element by element
    template<class X, class Y>
    Array1D<X> operator*(const Array1D<X> &vec1, const Array1D<X> &vec2) {
        Array1D<X> temp = vec1;
        temp *= vec2;
        return temp;
    }

    //! Divides two Array1Ds, element by element
    template<class X, class Y>
    Array1D<X> operator/(const Array1D<X> &vec1, const Array1D<X> &vec2) {
        Array1D<X> temp = vec1;
        temp /= vec2;
        return temp;
    }

    //! Takes cosine of array
    template<class X>
    Array1D<X> cos(const Array1D<X> &vec) {
        Array1D<X> temp = vec;
        temp.cos();
        return temp;
    }

    //! Takes sine of array
    template<class X>
    Array1D<X> sin(const Array1D<X> &vec) {
        Array1D<X> temp = vec;
        temp.sin();
        return temp;
    }

    /** Sets an Array1D to a constant
     */
    template<class X>
    Array1D<X> &Array1D<X>::operator=(const X &con) {
        Buffer1D<X>::operator=(con);
        return *this;
    }
};

#endif //ARRAY1D_H
