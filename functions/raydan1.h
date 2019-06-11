#ifndef RAYDAN1_H_INCLUDED
#define RAYDAN1_H_INCLUDED

#include <cmath>
#include "function.h"

namespace opt {
namespace function {

template<class real>
class raydan1 {
public:
    static real func(const la::vec<real>& v) {
        if (v.size() == 0) {
            throw "raydan1: n must be positive";
        }

        size_t n = v.size();
        real z = 0.0;

        for (size_t i = 0; i < n; ++i) {
            z += ((i+1) / 10.0) * (exp(v[i]) - v[i]);
        }

        return z;
    }

    static la::vec<real> gradient(const la::vec<real>& v) {
        if (v.size() == 0) {
            throw "raydan1: n must be positive";
        }

        size_t n = v.size();
        la::vec<real> z(n, 0.0);

        for (size_t i = 0; i < n; ++i) {
            z[i] = ((i+1) / 10.0) * (exp(v[i]) - 1);
        }

        return z;
    }

    static la::mat<real> hessian(const la::vec<real>& v) {
        if (v.size() == 0) {
            throw "raydan1: n must be positive";
        }

        size_t n = v.size();
        la::mat<real> z(n, n, 0.0);

        for (size_t i = 0; i < n; ++i) {
            z[i][i] = ((i+1) / 10.0) * exp(v[i]);
        }

        return z;
    }

    static la::vec<real> starting_point(const size_t n) {
        if (n == 0) {
            throw "raydan1: n must be positive";
        }

        return la::vec<real>(n, 1.0);
    }

    static function<real> getFunction() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};

}
}

#endif // RAYDAN1_H_INCLUDED
