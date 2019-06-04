#ifndef PROJEKATC___PP_QUAD_H
#define PROJEKATC___PP_QUAD_H

#include "function.h"

namespace opt {
namespace function {

template<class real>
class pp_quad {
public:
    static real func(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "pp_quad: n must be positive";
        real z = 0;
        z += v[0]*v[0];
        real ps = 0;
        for (size_t i=0; i<v.size(); i++) {
            ps += v[i];
            z += v[i]*v[i]*(i+1);
            z += ps*ps / 100;
        }
        return z;
    }

    static la::vec<real> gradient(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "pp_quad: n must be positive";
        la::vec<real> z(v.size(), 0.0);
        la::vec<real> ps(v.size());
        ps[0] = v[0];
        real t = 0;
        for (size_t i=1; i<v.size(); i++)
            ps[i] = ps[i-1] + v[i];
        for (size_t i=0; i<v.size(); i++)
            t += v[i] * (v.size() - i) * 2;

        z[0] = t / 100 + v[0] * 4;
        for (size_t i=1; i<v.size(); i++) {
            t -= 2*ps[i-1];
            z[i] = t / 100 + v[i] * (i+1) * 2;
        }
        return z;
    }

    static la::mat<real> hessian(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "pp_quad: n must be positive";
        la::mat<real> z(v.size(), v.size(), 0.0);
        for (size_t i=0; i<v.size(); i++) {
            for (size_t j=0; j<v.size(); j++) {
                if (i == j) {
                    if (i == 0)
                        z[i][j] += 200;
                    z[i][j] += 200*(i+1);
                }
                z[i][j] += 2*(v.size() - std::max(i, j));
            }
        }

        return z / (real)100;
    }

    static la::vec<real> starting_point(const size_t n) {
        if (n == 0)
            throw "pp_quad: n must be positive";
        return la::vec<real>(n, 0.5);
    }

    static function<real> getFunction() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};

}
}

#endif //PROJEKATC___PP_QUAD_H
