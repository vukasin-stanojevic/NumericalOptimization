#ifndef EXTENDED_HIMMELBLAU_H_INCLUDED
#define EXTENDED_HIMMELBLAU_H_INCLUDED

#include "function.h"

namespace opt {
namespace function {

template<class real>
class extended_himmelblau {
public:
    static real func(const la::vec<real>& v) {
        if (v.size() == 0) {
            throw "extended_himmelblau: n must be even and positive";
        }

        size_t n = v.size();
        real z = 0.0;

        for (size_t i = 0; i < n; i += 2) {
            real t = v[i]*v[i] + v[i+1] - 11;
            z += t*t;
            t = v[i] + v[i+1]*v[i+1] - 7;
            z += t*t;
        }

        return z;
    }

    static la::vec<real> gradient(const la::vec<real>& v) {
        if (v.size() == 0) {
            throw "extended_himmelblau: n must be even and positive";
        }

        size_t n = v.size();
        la::vec<real> z(n, 0.0);

        for (size_t i = 0; i < n; i += 2) {
            z[i] = 4*v[i]*v[i]*v[i] + 2*v[i+1]*v[i+1] + 4*v[i]*v[i+1] - 42*v[i] - 14;
            z[i+1] = 4*v[i+1]*v[i+1]*v[i+1] + 2*v[i]*v[i] + 4*v[i]*v[i+1] - 26*v[i+1] - 22;
        }

        return z;
    }

    static la::mat<real> hessian(const la::vec<real>& v) {
        if (v.size() == 0) {
            throw "extended_himmelblau: n must be even and positive";
        }

        size_t n = v.size();
        la::mat<real> z(n, n, 0.0);

        for (size_t i = 0; i < n; i += 2) {
            z[i][i] = 12*v[i]*v[i] + 4*v[i+1] - 42;
            z[i+1][i] = z[i][i+1] = 4*(v[i+1]+v[i]);
            z[i+1][i+1] = 12*v[i+1]*v[i+1] + 4*v[i] - 22;
        }

        return z;
    }

    static la::vec<real> starting_point(const size_t n) {
        if (n == 0) {
            throw "extended_himmelblau: n must be even and positive";
        }

        return la::vec<real>(n, 1.0);
    }

    static function<real> getFunction() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};

}
}

#endif // EXTENDED_HIMMELBLAU_H_INCLUDED
