#ifndef PROJEKATC___GRADIENT_DESCENT_H
#define PROJEKATC___GRADIENT_DESCENT_H

#include "base_method.h"

namespace opt {
namespace method {
namespace gradient {

template<class real>
class gradient_descent : public base_method<real> {
public:
    gradient_descent() : base_method<real>() {}
    gradient_descent(real epsilon) : base_method<real>(epsilon) {}
    gradient_descent(real epsilon, size_t max_iter) : base_method<real>(epsilon, max_iter) {}
    gradient_descent(real epsilon, size_t max_iter, real working_precision) : base_method<real>(epsilon, max_iter, working_precision) {}

    void operator()(function::function<real>& f, line_search::base_line_search<real>& ls, la::vec<real>& x) {
        this->iter_count = 0;
        this->tic();

        real f_curr = f(x);
        real f_prev = f_curr + 1; // should always pass the first working precision condition

        la::vec<real> gr = f.gradient(x);

        while (la::norm(gr) > this->epsilon && this->iter_count < this->max_iter && fabs(f_prev-f_curr)/(1+fabs(f_curr)) > this->working_precision) {
            ++this->iter_count;
            ls.push_f_val(f_curr);
            ls.set_current_f_val(f_curr);
            ls.set_current_g_val(gr);

            la::vec<real> d = -gr; // set direction to negative gradient
            x += d * ls(f, x, d); // move x in accordance with the line search

            f_prev = f_curr;
            f_curr = ls.get_current_f_val();
            gr = ls.get_current_g_val();
        }

        this->toc();
        this->f_min = f_curr;
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
