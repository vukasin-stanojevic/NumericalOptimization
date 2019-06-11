#ifndef NUMERICALOPTIMIZATION_GENERALIZED_PSC1_H
#define NUMERICALOPTIMIZATION_GENERALIZED_PSC1_H

#include "function.h"

namespace opt {
namespace function {

template<class real>
class gen_psc1 {
public:
    static real func(const la::vec<real>& v) {
        if (v.size() < 2)
            throw "gen_psc1: n must be greater than 1";
        real z = 0;
        for (size_t i=0; i<v.size()-1; ++i) {
            real t = v[i]*v[i] + v[i+1]*v[i+1] + v[i]*v[i+1];
            z += t*t;
            t = sin(v[i]);
            z += t*t;
            t = cos(v[i]);
            z += t*t;
        }
        return z;
    }

    static la::vec<real> gradient(const la::vec<real>& v) {
        if (v.size() < 2)
            throw "gen_psc1: n must be greater than 1";
        la::vec<real> z(v.size(), 0.0);
        auto n = v.size();
        real prev = 0;
        z[0] = 2*(v[0]*v[0] + v[1]*v[1] + v[0]*v[1])*(2*v[0] + v[1]);
        z[n-1] = 2*(v[n-2]*v[n-2] + v[n-1]*v[n-1] + v[n-2]*v[n-1])*(2*v[n-2] + v[n-1]);
        for (size_t i=1; i<n-1; ++i) {
            real t1 = v[i]*v[i] + v[i+1]*v[i+1] + v[i]*v[i+1];
            real t2 = v[i-1]*v[i-1] + v[i]*v[i] + v[i-1]*v[i];
            z[i] += 2*t1*(2*v[i] + v[i+1]) + 2*t2*(2*v[i] + v[i-1]);
        }
        return z;
    }

    static la::mat<real> hessian(const la::vec<real>& v) {
        if (v.size() < 2)
            throw "gen_psc1: n must be greater than 1";
        la::mat<real> z(v.size(), v.size(), 0.0);

        for (size_t i=0; i<v.size(); ++i) {
            // d^2 vi

            z[i][i] = 2*( (2*v[i] + v[i+1])*(2*v[i] + v[i+1]) + 2*(v[i]*v[i] + v[i+1]*v[i+1] + v[i]*v[i+1]) );

            if(i+1<v.size()){
                // d vivi+1
                z[i+1][i] = 2*( (2*v[i] + v[i+1])*(2*v[i+1] + v[i]) + (v[i]*v[i] + v[i+1]*v[i+1] + v[i]*v[i+1]) );
                // d vi+1vi
                z[i][i+1] = 2*( (2*v[i] + v[i+1])*(2*v[i+1] + v[i]) + (v[i]*v[i] + v[i+1]*v[i+1] + v[i]*v[i+1]) );
            }
        }

        return z;
    }

    static la::vec<real> starting_point(const size_t n) {
        if (n < 2)
            throw "gen_psc1: n must be greater than 1";
        la::vec<real> z(n, 0);
        for (size_t i=0; i<n; ++i) {
            if (i % 2) {
                z[i] = 0.1;
            } else {
                z[i] = 3;
            }
        }
        return z;
    }

    static function<real> getFunction() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};

}
}

#endif //NUMERICALOPTIMIZATION_GENERALIZED_PSC1_H
