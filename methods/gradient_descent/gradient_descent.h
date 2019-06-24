#ifndef PROJEKATC___GRADIENT_DESCENT_H
#define PROJEKATC___GRADIENT_DESCENT_H

#include "../base_method.h"

namespace opt {
namespace method {
namespace gradient {

template<class real>
class gradient_descent : public base_method<real> {
public:
    gradient_descent() : base_method<real>() {}
    void operator()(function::function<real>& f, line_search::base_line_search<real>& ls, method_params<real>& params) {
        this->tic();
        la::vec<real>& x = params.stariting_point;
        la::vec<real> gr = f.gradient(x);
        while (la::norm(gr) > params.epsilon && this->iter_count < params.max_iterations) {
            ++this->iter_count;

            la::vec<real> d = -gr;
            x += d * ls(f, x, d);

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

#endif //PROJEKATC___GRADIENT_DESCENT_H
