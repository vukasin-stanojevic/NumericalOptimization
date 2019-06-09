#ifndef PROJEKATC___WOLFE_H
#define PROJEKATC___WOLFE_H

#include <cmath>
#include "base_line_search.h"

namespace opt {
namespace line_search {

template<class real>
class wolfe : public base_line_search<real> {
private:
    real steepness; // rho
    real initial_step;
    real sigma;
    real xi;
    real max_step;
    real step_factor;
public:
    wolfe(std::map<std::string, real>& params){
        std::map<std::string, real> p;
        p["steepness"] = 1e-4;
        p["initial_step"] = 1;
        p["sigma"] = 0.9;
        p["xi"] = 1e-3;
        p["max_step"] = 1e10;
        p["step_factor"] = 10;
        this->rest(p, params);
        steepness = p["steepness"];
        initial_step = p["initial_step"];
        sigma = p["sigma"];
        xi = p["xi"];
        max_step = p["max_step"];
        step_factor = p["step_factor"];
        params = p;
    }

    real operator()(function::function<real>& func, la::vec<real>& x, la::vec<real>& d) {
        real a1 = 0, a2 = initial_step;
        real f0 = func(x);
        real f1 = f0;
        real f2 = func(x + d*a2);
        real pad0 = func.gradient(x).dot(d);
        real pad1 = pad0;
        real pad2 = func.gradient(x + d*a2).dot(d);

        size_t iter_num = 1;

        auto noc_zoom = [&]() {
            real a;
            while (1) {
                if (a1 < a2) {
                    a = this->cubic_interpolation(a1, a2, f1, f2, pad1, pad2);
                } else {
                    a = this->cubic_interpolation(a2, a1, f2, f1, pad2, pad1);
                }

                real ff = func(x + d*a);
                real pad = func.gradient(x + d*a).dot(d);

                if ((fabs(ff - f1) / (1 + fabs(ff)) < xi) ||
                    (fabs(ff - f2) / (1 + fabs(ff)) < xi)) {
                    return a;
                }

                if ((ff > f0 + steepness*a*pad0) || (ff >= f1)) {
                    // if we do not observe sufficient decrease in point a,
                    // we set the maximum of the feasible interval to a
                    a2 = a;
                    f2 = ff;
                    pad2 = pad;
                } else {
                    if (pad >= sigma * pad0) {
                        // strong wolfe fullfilled
                        return a;
                    }
                    a1 = a;
                    f1 = ff;
                    pad1 = pad;
                }
            }
        };

        while (1) {
            // armijo condition: check if current iteration violates sufficient decrease
            if (f2 > f0 + pad0*steepness*a2 || (f2 >= f1 && iter_num > 1)) {
                // there has to be an acceptable point between t0 and t1 because rho (steepness) > sigma
                return noc_zoom();
            }

            // current iterate has sufficient decrease, but are we too close?
            if (pad2 >= sigma*pad0) {
                // wolfe fullfilled, quit
                return a2;
            }

            // update values
            a1 = a2;
            f1 = f2;
            pad1 = pad2;

            a2 = fmin(a2*step_factor, max_step);
            f2 = func(x + d*a2);
            pad2 = func.gradient(x + d*a2).dot(d);

            ++iter_num;
        }
    }
};

}
}

#endif //PROJEKATC___WOLFE_H
