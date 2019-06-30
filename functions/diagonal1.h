#ifndef NUMERICALOPTIMIZATION_DIAGONAL1_H
#define NUMERICALOPTIMIZATION_DIAGONAL1_H

#include "function.h"

namespace opt {
namespace function {

template<class real>
class diagonal1 {
public:
    static real func(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "diagonal1: n must be positive";
        real z = 0;
        for (size_t i=0; i<v.size(); ++i) {
            real t = exp(v[i]) - (i+1)*v[i];
            z += t;
        }
        return z;
    }

    static la::vec<real> gradient(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "diagonal1: n must be positive";
        la::vec<real> z(v.size(), 0.0);
        for (size_t i=0; i<v.size(); ++i) {
            z[i] = exp(v[i]) - (i+1);
        }
        return z;
    }

    static la::mat<real> hessian(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "diagonal1: n must be positive";
        la::mat<real> z(v.size(), v.size(), 0.0);
        for (size_t i=0; i<v.size(); ++i) {
            z[i][i] = exp(v[i]);
        }

        return z;
    }

    static la::vec<real> starting_point(const size_t n) {
        if (n == 0)
            throw "diagonal1: n must be positive";
        return la::vec<real>(n,1.0/n);
    }

    static function<real> getFunction() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};

}
}

#endif //NUMERICALOPTIMIZATION_DIAGONAL1_H
