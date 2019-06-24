#ifndef PROJEKATC___FLETCHER_REEVES_H
#define PROJEKATC___FLETCHER_REEVES_H

#include <cmath>
#include "../base_method.h"

namespace opt {
namespace method {
namespace conjugate_gradient {

template<class real>
class fletcher_reeves : public base_method<real> {
public:
    fletcher_reeves() : base_method<real>() {}
    void operator()(function::function<real>& f, line_search::base_line_search<real>& ls, method_params<real>& params) {
        this->iter_count = 0;
        this->tic();

        la::vec<real>& x = params.stariting_point;

        real f_prev = f(x);
        la::vec<real> gr_old = f.gradient(x);

        la::vec<real> p0 = -gr_old;
        la::vec<real> s0 = p0;
        x += p0 * ls(f, x, p0);

        real f_curr = f(x);
        la::vec<real> gr = f.gradient(x);

        while (la::norm(gr) > params.epsilon && this->iter_count < params.max_iterations && fabs(f_prev - f_curr)/(1 + fabs(f_curr)) > params.working_precision) {
            ++this->iter_count;
            //std::cerr << this->iter_count << ": " << x << " grnorm = " << la::norm(gr) << '\n';

            real beta_fr = gr.dot(gr) / gr_old.dot(gr_old);

            // restart
            if (gr.dot(gr_old) / gr.dot(gr) > params.nu) {
                beta_fr = 0;
            }

            la::vec<real> p1 = -gr; // steepest direction at xn
            la::vec<real> s1 = p1 + s0 * beta_fr; // update the conjugate direction
            x += s1 * ls(f, x, s1);

            s0 = s1;
            f_prev = f_curr;
            f_curr = f(x);
            gr_old = gr;
            gr = f.gradient(x);
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

#endif //PROJEKATC___FLETCHER_REEVES_H
