#ifndef NUMERICALOPTIMIZATION_DFP_H
#define NUMERICALOPTIMIZATION_DFP_H

#include "../base_method.h"

namespace opt {
namespace method {
namespace quasi_newton {

template<class real>
class dfp : public base_method<real> {
public:
    dfp() : base_method<real>() {this->method_name = "DFP method";}
    dfp(real epsilon) : base_method<real>(epsilon) {this->method_name = "DFP method";}
    dfp(real epsilon, size_t max_iter) : base_method<real>(epsilon, max_iter) {this->method_name = "DFP method";}
    dfp(real epsilon, size_t max_iter, real working_precision) : base_method<real>(epsilon, max_iter, working_precision) {this->method_name = "DFP method";}

    void operator()(function::function<real>& f, line_search::base_line_search<real>& ls, la::vec<real>& x) {
        this->iter_count = 0;
        ls.clear_f_vals();
        this->gr_norms.clear();

        this->tic();

        size_t n = x.size();

        la::vec<real> x0;
        la::vec<real>& x1 = x;
        la::vec<real> gradient_prev;
        la::vec<real> gradient_curr = f.gradient(x1);

        la::mat<real> H = la::mat<real>::id(n);

        real fcur = f(x1);
        real fprev = fcur + 1;
        real gr_norm = la::norm(gradient_curr);
        this->gr_norms.push_back(gr_norm);

        while (gr_norm > this->epsilon && this->iter_count < this->max_iter && fabs(fprev-fcur)/(1+fabs(fcur)) > this->working_precision) {
            ++this->iter_count;
            ls.push_f_val(fcur);
            ls.set_current_f_val(fcur);
            ls.set_current_g_val(gradient_curr);

            la::vec<real> direction = (H.dot(gradient_curr))*(-1);

            fprev = fcur;
            x0 = x1;
            gradient_prev = gradient_curr;

            real t = ls(f, x1, direction);
            x1 += direction * t;

            fcur = ls.get_current_f_val();
            gradient_curr = ls.get_current_g_val();

            la::vec<real> s = x1 - x0;
            la::vec<real> y = gradient_curr - gradient_prev;

            la::vec<real> H_dot_y = H.dot(y);

            H +=  s.inner(s)/s.dot(s) - (H_dot_y).inner(H_dot_y)/(y.inner(H_dot_y));
            gr_norm = la::norm(gradient_curr);
            this->gr_norms.push_back(gr_norm);
        }

        this->toc();
        this->f_min = fcur;
        this->gr_norm = la::norm(gradient_curr);
        this->f_call_count = f.get_call_count();
        this->g_call_count = f.get_grad_count();
        this->h_call_count = f.get_hess_count();
    }
};

}
}
}


#endif //NUMERICALOPTIMIZATION_DFP_H
