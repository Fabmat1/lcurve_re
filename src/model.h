// Model and pparam definitions to closely follow what lcurve does

#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include "lcurve_base/array1d.h"
#include "lcurve_base/ldc.h"
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

#endif //MODEL_H
