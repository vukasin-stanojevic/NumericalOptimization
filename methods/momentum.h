//
// Created by lazar on 27.5.19..
//

#ifndef PROJEKATC___MOMENTUM_H
#define PROJEKATC___MOMENTUM_H

#include "base_method.h"

namespace opt_methods {
    namespace gradient {
        template<class real>
        class momentum : public base_method<momentum<real>, real> {
        public:
            template<class line_search,class function>
            void operator()(function &func,
                            line_search &lin_sr, la::vec<real> &x0) {
                auto p = -func.gradient(x0);
                while (norm(func.gradient(x0)) > 1e-7 && this->steps++ < 1000) {
                    cerr << x0 << "   gnorm = " << norm(func.gradient(x0)) << '\n';
                    p = p * 0.9 - func.gradient(x0) * 0.1;
                    x0 += p * lin_sr(x0, p, func);
                }
                cerr << "steps = " << this->steps << '\n';
            }
        };
    }
}


#endif //PROJEKATC___MOMENTUM_H
