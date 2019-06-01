//
// Created by lazar on 26.5.19..
//

#ifndef PROJEKATC___GRADIENT_DESCENT_H
#define PROJEKATC___GRADIENT_DESCENT_H

#include "base_method.h"

namespace opt_methods {
    namespace gradient {
        template<class real>
        class gradient_descent : public base_method<gradient_descent<real>, real> {
        public:
            template<class line_search,class function>
            void operator()(function &func,
                            line_search &lin_sr, la::vec<real> &x0) {
                while (norm(func.gradient(x0)) > 1e-8 && this->steps++ < 1000) {
                    cerr << x0 << "   grad: ";
                    cerr << func.gradient(x0) << '\n';
                    auto p = -func.gradient(x0);
                    x0 += p * lin_sr(x0, p, func);
                }
                cerr << "steps = " << this->steps << '\n';
            }
        };
    }
}

#endif //PROJEKATC___GRADIENT_DESCENT_H
