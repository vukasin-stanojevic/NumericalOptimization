//
// Created by lazar on 27.5.19..
//

#ifndef PROJEKATC___FLETCHER_REEVES_H
#define PROJEKATC___FLETCHER_REEVES_H


#include "base_method.h"

namespace opt_methods {
    namespace conjugate_gradient {
        template<class real>
        class fletcher_reeves : public base_method<fletcher_reeves<real>, real> {
        public:
            template<class line_search,class function>
            void operator()(function &func,
                            line_search &lin_sr, la::vec<real> &x0) {
                auto p0 = -func.gradient(x0);
                real a0 = lin_sr(x0, p0, func);
                auto x1 = x0 + p0*a0;
                auto s0 = p0;

                while (norm(func.gradient(x1)) > 1e-7 && this->steps++ < 100000) {
                    cerr << x1 << "   gnorm = " << norm(func.gradient(x1)) << '\n';
                    auto p1 = -func.gradient(x1); // steepest direction at xn
                    auto beta1 = x1.dot(x1) / x0.dot(x0); // FR beta
                    auto s1 = p1 + s0*beta1; // update the conjugate direction
                    auto a1 = lin_sr(x1, s1, func);

                    auto x2 = x1 + s1*a1;

                    x1 = x2;
                    s0 = s1;
                }
                x0 = x1;
                cerr << func(x0) << " " << norm(func.gradient(x1)) << endl;
                cerr << "steps = " << this->steps << '\n';
            }
        };
    }
}

#endif //PROJEKATC___FLETCHER_REEVES_H
