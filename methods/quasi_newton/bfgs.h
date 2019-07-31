#ifndef NUMERICALOPTIMIZATION_BFGS_H
#define NUMERICALOPTIMIZATION_BFGS_H

#include "../base_method.h"

namespace opt {
namespace method {
namespace quasi_newton {

template<class real>
class bfgs : public base_method<real> {
public:
    bfgs() : base_method<real>() {}
    bfgs(real epsilon) : base_method<real>(epsilon) {}
    bfgs(real epsilon, size_t max_iter) : base_method<real>(epsilon, max_iter) {}
    bfgs(real epsilon, size_t max_iter, real working_precision) : base_method<real>(epsilon, max_iter, working_precision) {}

    void operator()(function::function<real>& f, line_search::base_line_search<real>& ls, la::vec<real>& x) {
        this->iter_count = 0;
        ls.clear_f_vals();
        
        this->tic();

        size_t n = x.size();

        la::vec<real> x0;
        la::vec<real>& x1 = x;
        la::vec<real> gr0;
        la::vec<real> gr1 = f.gradient(x1);

        la::mat<real> H = la::mat<real>::id(n);

        real fcur = f(x1);
        real fprev = fcur + 1;

        while (la::norm(gr1) > this->epsilon && this->iter_count < this->max_iter && fabs(fprev-fcur)/(1+fabs(fcur)) > this->working_precision) {
            ++this->iter_count;
            ls.push_f_val(fcur);
            ls.set_current_f_val(fcur);
            ls.set_current_g_val(gr1);

            la::vec<real> direction = (H.dot(gr1))*(-1);

            fprev = fcur;
            x0 = x1;
            gr0 = gr1;

            real t = ls(f, x1, direction);
            x1 += direction * t;

            fcur = ls.get_current_f_val();
            gr1 = ls.get_current_g_val();

            la::vec<real> s = x1 - x0;
            la::vec<real> y = gr1 - gr0;

            real auxSc = s.dot(y); // auxiliary variable
            la::vec<real> H_dot_y = H.dot(y);

            H +=  s.outer(s)*(auxSc + y.dot(H_dot_y))/(auxSc*auxSc) - (H_dot_y.outer(s) + (s.outer(H_dot_y)))/auxSc;
        }

        this->toc();
        this->f_min = fcur;
        this->gr_norm = la::norm(gr1);
        this->f_call_count = f.get_call_count();
        this->g_call_count = f.get_grad_count();
        this->h_call_count = f.get_hess_count();
    }
};

}
}
}

#endif //NUMERICALOPTIMIZATION_BFGS_H
