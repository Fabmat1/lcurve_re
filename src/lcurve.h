//
// Created by fabian on 4/29/25.
//

#ifndef LCURVE_H
#define LCURVE_H

#include <string>
#include <vector>

using namespace std;

namespace Lcurve {

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
    istream& operator>>(std::istream& s, Datum& datum);

    //! ASCII output of a Datum (expects time expose flux ferr weight ndiv)
    ostream& operator<<(std::ostream& s, const Datum& datum);

    // Holds a light curve
    class Data : public vector<Datum> {

    public:

        //! Default constructor
        Data() : vector<Datum>() {}

        //! Constructor with pre-defined size
        Data(int n) : vector<Datum>(n) {}

        //! Constructor from a file
        Data(const string& file);

        //! Writes to an ASCII file
        void wrasc(const string& file) const;

        //! Reads from an ASCII file
        void rasc(const string& file);

    };
};



#endif //LCURVE_H
