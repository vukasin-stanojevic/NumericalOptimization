//
// Created by lazar on 26.5.19..
//

#ifndef PROJEKATC___GOLDSTEIN_H
#define PROJEKATC___GOLDSTEIN_H


#include "line_search.h"


namespace line_searches {
    template<class real>
    class goldstein : public line_search<real> {
    private:
        real steepness;
        real initial_step;
        real gamma;
    public:
        goldstein(map<string, real>& params){
            map<string,real> p;
            p["steepness"] = 1e-4;
            p["initial_step"] = 1;
            p["gamma"] = 1.1;
            rest(p,params);
            steepness = p["steepness"];
            initial_step = p["initial_step"];
            gamma = p["gamma"];
            params = p;

        }

        real operator()(vec<real> &x0, vec<real> &d, functions::function<real> &func) {
            real pad = func.gradient(x0).dot(d); // kojom brzinom raste f u pravcu d
            // drugim recima, to je priblizna vrednost f(x0+d) - f(x0)

            real a1 = 0, a2 = 0, a = initial_step;
            bool a2inf = true;
            real f0 = func(x0);
            real ff = func(x0 + d*a);
            size_t steps = 1;

            while (steps < 52) {
                if (ff > f0 + steepness*a*pad) {
                    a2 = a;
                    a2inf = false;
                    a = (a1 + a2) / 2;
                } else if (ff < f0 + (1-steepness)*a*pad) {
                    a1 = a;
                    if (!a2inf) {
                        a = (a1 + a2) / 2;
                    } else {
                        a *= gamma;
                    }
                } else {
                    break;
                }

                ++steps;
                ff = func(x0 + d*a);
            }

            return a;
        }
    };
}

#endif //PROJEKATC___GOLDSTEIN_H
