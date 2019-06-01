//
// Created by lazar on 26.5.19..
//

#ifndef PROJEKATC___BINARY_H
#define PROJEKATC___BINARY_H


#include "line_search.h"


namespace line_searches {
    template<class real>
    class binary : public line_search<real> {
    private:
        real steepness;
        real initial_step;
    public:
        binary(map<string, real>& params){
            map<string,real> p;
            p["initial_step"] = 1;
            rest(p,params);
            initial_step = p["initial_step"];
            params = p;

        }

        real operator()(vec<real> &x0, vec<real> &d, functions::function<real> &func) {
            real a = initial_step;
            // real fstart = f(x0);
            real f0 = func(x0 + d * a);
            real f1 = func(x0 + d * a * 2);

            if (f1 < f0) {
                a *= 2;
                real curr = f1;
                real t = func(x0 + d*a*2);
                while (t < curr) {
                    curr = t;
                    a *= 2;
                    t = func(x0 + d*a*2);
                }
                return a;
            } else {
                real curr = f0;
                real t = func(x0 + d*a/2);
                while (t < curr) {
                    a /= 2;
                    curr = t;
                    t = func(x0 + d*a/2);
                }
                return a;
            }
        }
    };
}

#endif //PROJEKATC___BINARY_H
