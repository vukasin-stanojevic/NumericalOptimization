#ifndef PROJEKATC___FULL_HESSIAN_FH2_H
#define PROJEKATC___FULL_HESSIAN_FH2_H

#include "function.h"

namespace opt {
namespace function {

template<class real>
class full_hessian_fh2 {
public:
    static real func(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "full_hessian_fh2: n must be positive";
        real z = 0, ps = 0;
        z = (v[0] - 5);
        z *= z;
        ps = v[0];
        for (size_t i=1; i<v.size(); i++) {
            ps += v[i];
            z += (ps - 1) * (ps - 1);
        }
        return z;
    }

    static la::vec<real> gradient(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "full_hessian_fh2: n must be positive";
        // return value
        la::vec<real> z(v.size(), 0.0);
        // prefix sums
        la::vec<real> ps(v.size(), 0.0);
        ps[0] = v[0];
        for (size_t i=1; i<v.size(); i++)
            ps[i] = ps[i-1] + v[i];

        // starting result
        real t = -2*(real)v.size();
        for (size_t i=0; i<v.size(); i++)
            t += (v.size() - i) * v[i] * 2;
        z[0] = t - 8;
        for (size_t i=1; i<v.size(); i++) {
            t -= 2*ps[i-1];
            t += 2;
            z[i] = t;
        }
        return z;
    }

    static la::mat<real> hessian(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "full_hessian_fh2: n must be positive";
        la::mat<real> z(v.size(), v.size(), 0.0);
        for (size_t i=0; i<v.size(); i++)
            for (size_t j=0; j<v.size(); j++)
                z[i][j] = 2*(v.size() - std::max(i, j));
        return z;
    }

    static la::vec<real> starting_point(const size_t n) {
        if (n == 0)
            throw "full_hessian_fh2: n must be positive";
        return la::vec<real>(n, 0.01);
    }

    static function<real> getFunction() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};

}
}

#endif //PROJEKATC___FULL_HESSIAN_FH2_H
