//
// Created by lazar on 26.5.19..
//

#ifndef PROJEKATC___ARMIJO_H
#define PROJEKATC___ARMIJO_H


#include "line_search.h"


namespace line_searches {
    template<class real>
    class armijo : public line_search<real> {
    private:
        real steepness;
        real initial_step;
    public:
        armijo(map<string, real>& params){
            map<string,real> p;
            p["steepness"] = 1e-4;
            p["initial_step"] = 1;
            rest(p,params);
            steepness = p["steepness"];
            initial_step = p["initial_step"];
            params = p;

        }
        real operator()(vec<real> &x0, vec<real> &d, functions::function<real> &func) {
            real f0 = func(x0); // vrednost u polaznoj tacki

            real pad = func.gradient(x0).dot(d); // kojom brzinom raste f u pravcu d
            // drugim recima, to je priblizna vrednost f(x0+d) - f(x0)

            real a_curr = initial_step; // nalazimo se u tacki x0 + d*a

            real f_curr, f_prev, a_prev;
            f_curr = func(x0 + d * a_curr);

            size_t steps = 1;

            while (f_curr > f0 - steepness * a_curr * pad) {
                real a_new;
                if (steps == 1) {
                    // nadji sledecu tacku kvadratnom interpolacijom
                    a_new = pad * a_curr * a_curr / 2 / (f0 - f_curr + pad * a_curr);
                } else {
                    // nadji kubnom interpolacijom
                    real cubic = a_prev * a_prev * (f_curr - f0);
                    cubic -= a_curr * a_prev * a_prev * pad;
                    cubic += a_curr * a_curr * (f0 - f_prev + a_prev * pad);
                    cubic /= a_curr * a_curr * (a_curr - a_prev) * a_prev * a_prev;

                    real quadr = -cubic * a_curr * a_curr * a_curr - f0 + f_curr - a_curr * pad;
                    quadr /= a_curr * a_curr;

                    a_new = (-quadr + sqrt(quadr * quadr - 3 * cubic * pad)) / (3 * cubic);
                }

                a_prev = a_curr;
                a_curr = a_new;

                f_prev = f_curr;
                f_curr = func(x0 + d * a_curr);

                ++steps;
            }

            return a_curr;
        }
    };
}

#endif //PROJEKATC___ARMIJO_H
