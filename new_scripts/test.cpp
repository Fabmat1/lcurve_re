#include "../src/new_helpers.h"


int main(){
    double r1 = 0.2, r1_err = 0.01;
    cout << Helpers::compute_scaled_r1(r1+r1_err, 200, 0.25) << " ";
    cout << Helpers::compute_scaled_r1(r1-r1_err, 200,  0.25) << endl;
}