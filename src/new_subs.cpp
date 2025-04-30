#include "new_subs.h"
#include <sstream>
#include <iomanip>
#include <string>


Subs::Format::Format(int precision) :
  precision_(precision), width_(0), format_(std::ios::fmtflags(0)), upper(false),
  showpnt(true), fill_char(' '), fadjust(std::ios::left) {}


void Subs::Format::precision(int p) {
  precision_ = p;
}

void Subs::Format::scientific() {
  format_ = std::ios::scientific;
}

void Subs::Format::fixed(){
  format_ = std::ios::fixed;
}

void Subs::Format::general(){
  format_ = std::ios::fmtflags(0);
}

void Subs::Format::width(int w){
  width_  = w;
}

void Subs::Format::showpoint(){
  showpnt = true;
}

void Subs::Format::noshowpoint(){
  showpnt = false;
}

void Subs::Format::uppercase() {
  upper = true;
}

void Subs::Format::lowercase(){
  upper = false;
}

void Subs::Format::fill(char c){
  fill_char = c;
}

void Subs::Format::left(){
  fadjust = std::ios::left;
}


void Subs::Format::right(){
  fadjust = std::ios::right;
}

void Subs::Format::internal(){
  fadjust = std::ios::internal;
}


double Subs::svdfit(const Buffer1D<ddat>& data, Buffer1D<double>& a, const Buffer2D<double>& vect,
        Buffer2D<double>& u, Buffer2D<double>& v, Buffer1D<double>& w){

  if(a.size() != vect.get_nx())
    throw Subs_Error("svdfit[double]: number of coefficients = " + to_string(a.size()) +
         " in parameter vector does not match number in function array = " + to_string(vect.get_nx()));
  if(data.size() != vect.get_ny())
    throw Subs_Error("svdfit[double]: number of data = " + to_string(data.size()) +
         " does not match number in function array = " + to_string(vect.get_ny()));

  size_t ndata = data.size();
  size_t nc    = a.size();
  size_t ndat = 0;
  size_t i, j, k;
  for(j=0; j<ndata; j++)
    if(data[j].z>0.) ndat++;

  u.resize(ndat,nc);
  v.resize(nc,nc);
  w.resize(nc);

  const double TOL = 1.e-5;
  double tmp;

  Buffer1D<double> b(ndat);
  for(i=0,k=0; i<ndata; i++){
    if(data[i].z>0.){
      tmp = 1./data[i].z;
      for(j=0; j<nc; j++)
        u[k][j] = tmp*vect[i][j];
      b[k] = tmp*data[i].y;
      k++;
    }
  }

  svdcmp(u, w, v);

  // Edit singular values
  double wmax = 0.;
  for(i=0; i<nc; i++)
    if(w[i] > wmax) wmax = w[i];

  double thresh = TOL*wmax;
  for(i=0; i<nc; i++)
    if(w[i] < thresh) w[i] = 0.;

  // Carry on
  svbksb(u, w, v, b, a);

  double sum, chisq = 0.;

  for(i=0,k=0; i<ndata; i++){
    if(data[i].z>0.){
      for(j=0, sum=0.; j<nc; j++)
        sum += a[j]*vect[i][j];
      chisq += ((data[i].y-sum)/data[i].z)*((data[i].y-sum)/data[i].z);
    }
  }
  return chisq;
}

// //! Extractor for doubles
// std::ostream& Subs::operator<<(std::ostream& ostr, const Subs::Bound_form_d& bf){
//   std::ostringstream  s;
//   s.precision(bf.form.precision_);
//   s.width(bf.form.width_);
//   s.fill(bf.form.fill_char);
//   s.setf(bf.form.format_, std::ios::floatfield);
//   s.setf(bf.form.fadjust, std::ios::adjustfield);
//   if(bf.form.upper) s.setf(std::ios::uppercase);
//   s << bf.val;
//   return ostr << s.str();
// }
//
// //! Extractor for floats
// std::ostream& Subs::operator<<(std::ostream& ostr, const Subs::Bound_form_f& bf){
//   std::ostringstream  s;
//   s.precision(bf.form.precision_);
//   s.width(bf.form.width_);
//   s.fill(bf.form.fill_char);
//   s.setf(bf.form.format_, std::ios::floatfield);
//   s.setf(bf.form.fadjust, std::ios::adjustfield);
//   if(bf.form.upper) s.setf(std::ios::uppercase);
//   if(bf.form.showpnt) s.setf(std::ios::showpoint);
//   s << bf.val;
//   return ostr << s.str();
// }
//
// //! Extractor for strings
// std::ostream& Subs::operator<<(std::ostream& ostr, const Subs::Bound_form_s& bf){
//   std::ostringstream s;
//   s.width(bf.form.width_);
//   s.fill(bf.form.fill_char);
//   s.setf(bf.form.fadjust, std::ios::adjustfield);
//   if(bf.form.upper) s.setf(std::ios::uppercase);
//   s << bf.val;
//   return ostr << s.str();
// }
//
// //! Returns combination of format and double object
// Subs::Bound_form_d Subs::Format::operator()(double d) const {
//   return Bound_form_d(*this, d);
// }
//
// //! Returns combination of format and float object
// Subs::Bound_form_f Subs::Format::operator()(float f) const {
//   return Bound_form_f(*this, f);
// }
//
// //! Returns combination of format and string object
// Subs::Bound_form_s Subs::Format::operator()(const std::string& s) const {
//     return Bound_form_s(*this, s);
// }
//
// //! Returns combination of format and string object
// Subs::Bound_form_s Subs::Format::operator()(const std::string& s, int width) {
//     this->width(width);
//     return Bound_form_s(*this, s);
// }


















