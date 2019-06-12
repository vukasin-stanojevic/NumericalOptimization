#ifndef PROJEKATC___MOMENTUM_H
#define PROJEKATC___MOMENTUM_H

#include "base_method.h"

namespace opt {
namespace method {
namespace gradient {

template<class real>
class momentum : public base_method<real> {
public:
    momentum() : base_method<real>() {}
    void operator()(function::function<real>& f, line_search::base_line_search<real>& ls, la::vec<real>& x) {
        this->tic();

        la::vec<real> gr = f.gradient(x);
        la::vec<real> p = -gr;
        while (la::norm(gr) > 1e-7 && this->iter_count++ < 10000) {
            p = p * 0.9 - gr * 0.1;
            x += p * ls(f, x, p);

            gr = f.gradient(x);
        }

        this->toc();
        this->f_min = f(x);
        this->gr_norm = la::norm(gr);
        this->f_call_count = f.get_call_count();
        this->g_call_count = f.get_grad_count();
        this->h_call_count = f.get_hess_count();
    }
};

}
}
}

#endif //PROJEKATC___MOMENTUM_H
