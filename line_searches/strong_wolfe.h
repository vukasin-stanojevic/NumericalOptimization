//
// Created by lazar on 26.5.19..
//

#ifndef PROJEKATC___STRONG_WOLFE_H
#define PROJEKATC___STRONG_WOLFE_H


#include "line_search.h"


namespace line_searches {
    template<class real>
    class strong_wolfe : public line_search<real> {
    private:
        real steepness;
        real initial_step;
        real sigma;
        real xi;
        real max_step;
        real step_factor;
    public:
        strong_wolfe(map<string, real>& params){
            map<string,real> p;
            p["steepness"] = 1e-4;
            p["initial_step"] = 1;
            p["sigma"] = 0.1; // strong!
            p["xi"] = 1e-3;
            p["max_step"] = 1e10;
            p["step_factor"] = 10;
            rest(p,params);
            steepness = p["steepness"];
            initial_step = p["initial_step"];
            sigma = p["sigma"];
            xi = p["xi"];
            max_step = p["max_step"];
            step_factor = p["step_factor"];
            params = p;

        }

        real operator()(vec<real> &x0, vec<real> &d, functions::function<real> &func) {
            real a1 = 0, a2 = initial_step;
            real f0 = func(x0);
            real f1 = f0;
            real f2 = func(x0 + d*a2);
            real pad0 = func.gradient(x0).dot(d);
            real pad1 = pad0;
            real pad2 = func.gradient(x0 + d*a2).dot(d);

            size_t steps = 1;

            auto noc_zoom = [&]() {
                double a = 0.0;
                while (1) {
                    if (a1 < a2)
                        a = interp(a1, a2, f1, f2, pad1, pad2);
                    else
                        a = interp(a2, a1, f2, f1, pad2, pad1);

                    real ff = func(x0 + d*a);
                    real pad = func.gradient(x0 + d*a).dot(d);

                    if ((abs(ff - f1) / (1 + abs(ff)) < xi) ||
                        (abs(ff - f2) / (1 + abs(ff)) < xi))
                    {
                        return a;
                    }

                    if ((ff > f0 + steepness*a*pad0) || (ff >= f1)) {
                        a2 = a;
                        f2 = ff;
                        pad2 = pad;
                    } else {
                        if (abs(pad1) <= -sigma*pad0)
                            return a;
                        if (pad1*(a2-a1) >= 0) {
                            a2 = a1;
                            f2 = f1;
                            pad2 = pad1;
                        }
                        a1 = a;
                        f1 = ff;
                        pad1 = pad;
                    }
                }
                return a;
            };

            while (1) {
                // <----- armijo uslov ----->
                if (f2 > f0 + pad0*steepness*a2 || (f1 <= f2 && steps > 1))
                    return noc_zoom();

                // strong!!!
                if (abs(pad2) <= -sigma*pad1)
                    return a2;

                // strong!
                if (pad2 >= 0)
                    return noc_zoom();

                a1 = a2;
                f1 = f2;
                pad1 = pad2;

                a2 = min(max_step, a2*step_factor);
                f2 = func(x0 + d*a2);
                pad2 = func.gradient(x0 + d*a2).dot(d);
                ++steps;
            }
        }
    };
}


#endif //PROJEKATC___STRONG_WOLFE_H
