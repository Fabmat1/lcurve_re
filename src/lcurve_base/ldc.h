// LDC class from Tom Marsh

#ifndef LDC_H
#define LDC_H

//! Lim darkening class
class LDC {
public:
    enum LDCtype { POLY, CLARET };

    //! Default. Sets all to zero.
    LDC() : ldc1(0.), ldc2(0.), ldc3(0.), ldc4(0.), mucrit(0.), ltype(POLY) {
    }

    //! Standard constructor
    LDC(double ldc1, double ldc2, double ldc3, double ldc4, double mucrit, LDCtype ltype) : ldc1(ldc1), ldc2(ldc2),
        ldc3(ldc3), ldc4(ldc4), mucrit(mucrit), ltype(ltype) {
    }

    //! Computes I(mu)
    double imu(double mu) const {
        if (mu <= 0) {
            return 0.;
        } else {
            mu = std::min(mu, 1.);
            double ommu = 1. - mu, im = 1.;
            if (this->ltype == POLY) {
                im -= ommu * (this->ldc1 + ommu * (this->ldc2 + ommu * (this->ldc3 + ommu * this->ldc4)));
            } else if (this->ltype == CLARET) {
                im -= this->ldc1 + this->ldc2 + this->ldc3 + this->ldc4;
                double msq = sqrt(mu);
                im += msq * (this->ldc1 + msq * (this->ldc2 + msq * (this->ldc3 + msq * this->ldc4)));
            }
            return im;
        }
    }

    //! To help applying mucrit
    bool see(double mu) const { return mu > this->mucrit; }

private:
    double ldc1;
    double ldc2;
    double ldc3;
    double ldc4;
    double mucrit;
    LDCtype ltype;
};
#endif //LDC_H
