#ifndef HESTENES_STIEFEL_H_INCLUDED
#define HESTENES_STIEFEL_H_INCLUDED

#include <cmath>
#include "../base_method.h"

namespace opt {
namespace method {
namespace conjugate_gradient {

template<class real>
class hestenes_stiefel : public base_method<real> {
public:
    hestenes_stiefel() : base_method<real>(), nu(0.1) {}
    hestenes_stiefel(real nu) : base_method<real>(), nu(nu) {}
    hestenes_stiefel(real nu, real epsilon) : base_method<real>(epsilon), nu(nu) {}
    hestenes_stiefel(real nu, real epsilon, size_t max_iter) : base_method<real>(epsilon, max_iter), nu(nu) {}
    hestenes_stiefel(real nu, real epsilon, size_t max_iter, real working_precision) : base_method<real>(epsilon, max_iter, working_precision), nu(nu) {}

    void operator()(function::function<real>& f, line_search::base_line_search<real>& ls, la::vec<real>& x) {
        this->iter_count = 0;
        ls.clear_f_vals();
        
        this->tic();

        real f_curr = f(x);
        real f_prev = f_curr + 1;

        la::vec<real> gr = f.gradient(x);
        la::vec<real> gr_old;

        la::vec<real> pk = -gr;

        while (la::norm(gr) > this->epsilon && this->iter_count < this->max_iter && fabs(f_prev-f_curr)/(1+fabs(f_curr)) > this->working_precision) {
            ++this->iter_count;
            ls.push_f_val(f_curr);
            ls.set_current_f_val(f_curr);
            ls.set_current_g_val(gr);

            f_prev = f_curr;
            gr_old = gr;

            x += pk * ls(f, x, pk);

            f_curr = ls.get_current_f_val();
            gr = ls.get_current_g_val();

            real beta_hs = gr.dot(gr - gr_old) / pk.dot(gr - gr_old);

            // if restart coefficient greater than nu, apply reset
            real rc = fabs(gr.dot(gr_old)) / gr.dot(gr);
            if (rc > nu) {
                beta_hs = 0;
            }

            pk *= beta_hs;
            pk -= gr;
        }

        this->toc();
        this->f_min = f_curr;
        this->gr_norm = la::norm(gr);
        this->f_call_count = f.get_call_count();
        this->g_call_count = f.get_grad_count();
        this->h_call_count = f.get_hess_count();
    }
protected:
    real nu; // used for conditionally restarting beta
};

}
}
}

#endif //HESTENES_STIEFEL_H_INCLUDED
