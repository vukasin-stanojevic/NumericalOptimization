#ifndef NUMERICALOPTIMIZATION_AP_QUAD_H
#define NUMERICALOPTIMIZATION_AP_QUAD_H

#include "function.h"

namespace opt {
namespace function {

template<class real>
class almost_pert_quad {
public:
    static real func(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "almost_pert_quad: n must be positive";
        real z = 0;
        size_t n = v.size();
        for (size_t i=0; i<n; ++i) {
            real t = (i+1)*v[i]*v[i];
            z += t;
        }
        z+=(v[0]+v[n-1])*(v[0]+v[n-1])/100;
        return z;
    }

    static la::vec<real> gradient(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "almost_pert_quad: n must be positive";
        size_t n = v.size();
        la::vec<real> z(n, 0.0);
        for (size_t i=0; i<n; ++i) {
            z[i] = 2*(i+1)*v[i];
        }
        z[0]+= (v[0]+v[n-1])/50.0;
        z[n-1]+= (v[0]+v[n-1])/50.0;
        return z;
    }

    static la::mat<real> hessian(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "almost_pert_quad: n must be positive";
        la::mat<real> z(v.size(), v.size(), 0.0);
        size_t n = v.size();
        for (size_t i=0; i<n; ++i) {
            z[i][i] = 2*(i+1);
        }
        z[n-1][n-1] += 1/50.0;
        z[0][0] += 1/50.0;
        z[n-1][0] += 1/50.0;
        z[0][n-1] += 1/50.0;

        return z;
    }

    static la::vec<real> starting_point(const size_t n) {
        if (n == 0)
            throw "almost_pert_quad: n must be positive";
        return la::vec<real>(n,0.5);
    }

    static function<real> getFunction() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};

}
}

#endif //NUMERICALOPTIMIZATION_AP_QUAD_H
