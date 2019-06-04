#ifndef NUMERICALOPTIMIZATION_FLETCHCR_H
#define NUMERICALOPTIMIZATION_FLETCHCR_H

#include "function.h"

namespace opt {
namespace function {

template<class real>
class fletchcr {
public:
    static real func(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "fletchcr: n must be positive";
        real z = 0;
        real c = 100;

        for (size_t i=0; i<v.size()-1; i++) {
            real t = (v[i+1] - v[i] + 1 - v[i]*v[i]);
            z+= c*t*t;
        }
        return z;
    }

    static la::vec<real> gradient(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "fletchcr: n must be positive";

        real c = 100;
        auto n = v.size();
        la::vec<real> z(v.size(), 0.0);
        z[0] = -2*c*(2*v[0]+1)*(v[1] - v[0] + 1 - v[0]*v[0]);
        z[n-1] = 2*c*(v[n-1] - v[n-2] + 1 - v[n-2]*v[n-2]);
        for(int i=1;i<n-1;++i){
            z[i] = -2*c*(2*v[i]+1)*(v[i+1] - v[i] + 1 - v[i]*v[i]) + 2*c*(v[i] - v[i-1] + 1 - v[i-1]*v[i-1]);
        }
        return z;
    }

    static la::mat<real> hessian(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "fletchcr: n must be positive";
        la::mat<real> z(v.size(), v.size(), 0.0);

        auto n = v.size();
        real c = 100;


        for (size_t i=0; i<v.size()-1; i++){
            real s1 = 2*v[i]+1;
            real s2 = v[i+1] - v[i] + 1 - v[i]*v[i];
            z[i][i] = 2*c*(s1*s1 - 2* s2) + 2*c;

            z[i][i+1] = -2*c*(2*v[i] + 1);
            z[i+1][i] = -2*c*(2*v[i] + 1);

        }
        z[n-1][n-1] = 2*c;

        return z;
    }

    static la::vec<real> starting_point(const size_t n) {
        if (n == 0)
            throw "fletchcr: n must be positive";
        return la::vec<real>(n, 0.0);
    }

    static function<real> getFunction() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};

}
}

#endif //NUMERICALOPTIMIZATION_FLETCHCR_H
