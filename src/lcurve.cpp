//
// Created by fabian on 4/29/25.
//

#include "lcurve.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include "new_subs.h"

/** Reads in data from a file in the form of a series of lines
 * specifying each 'Datum'. Lines starting with # or blank are ignored
 * \param file the ASCII file to load.
 */
Lcurve::Data::Data(const std::string& file) : std::vector<Datum>() {
    this->rasc(file);
}

std::istream& Lcurve::operator>>(std::istream& s, Datum& datum){
    std::string buff;
    getline(s, buff);
    std::istringstream istr(buff);
    istr >> datum.time >> datum.expose >> datum.flux >> datum.ferr >>
        datum.weight >> datum.ndiv;
    if(!istr)
        throw runtime_error("Lcurve::operator>>: failed to read t,e,f,fe,wgt,ndiv of a Datum");
    if(datum.ferr < 0.){
        datum.ferr -= datum.ferr;
        datum.weight = 0.;
    }
    return s;
}

/** ASCII output of a Datum.
 */
std::ostream& Lcurve::operator<<(std::ostream& s, const Datum& datum) {
    s << std::setw(17) << std::fixed << std::setprecision(8) << datum.time << " "
      << datum.expose << " "
      << std::setw(10) << std::fixed << std::setprecision(5) << datum.flux << " "
      << datum.ferr << " "
      << datum.weight << " "
      << datum.ndiv;
    return s;
}

/** Reads in data from a file in the form of a series of lines
 * specifying each 'Datum'. Lines starting with # or blank are ignored
 * \param file the ASCII file to load.
 */
void Lcurve::Data::rasc(const std::string& file) {

    // Read in the parameter values
    std::ifstream fin(file.c_str());
    if(!fin)
        throw runtime_error("Lcurve::Data::Data: failed to open " + file +
                           " for data.");

    this->clear();

    const int MAX_LINE = 5000;
    int n = 0;
    char ch;
    Datum datum;
    while(fin && !fin.eof()){
        n++;
        ch = fin.peek();
        if(fin.eof()){
            std::cout << "End of file reached." << std::endl;
        }else if(ch == '#' || ch == ' ' || ch == '\t' || ch == '\n'){
            fin.ignore(MAX_LINE, '\n'); // skip to next line
        }else{
            if(fin >> datum){
                this->push_back(datum);
            }else if(fin.eof()){
                std::cout << "End of data file reached." << std::endl;
            }else{
                throw runtime_error("Data file input failure on line " +
                                   to_string(n));
            }
        }
    }
    fin.close();

    std::cout << this->size() << " lines of data read from " << file
              << "\n" << std::endl;

}

/** Writes to an ASCII file with a series of rows
 * each with a time, exposure, flux and uncertainty
 */
void Lcurve::Data::wrasc(const std::string& file) const {

    std::ofstream fout(file.c_str());
    if(!fout)
        throw runtime_error("Lcurve::Data::wrasc: failed to open " +
                           file + " for output.");

    for(const auto & i : *this)
        fout << i << std::endl;

}