#ifndef CG_DESCENT_H_INCLUDED
#define CG_DESCENT_H_INCLUDED

#include <cmath>
#include "../base_method.h"

namespace opt {
namespace method {
namespace conjugate_gradient {

template<class real>
class cg_descent : public base_method<real> {
public:
    cg_descent(real eta = 0.01, real delta = 0.7) : base_method<real>(), eta(eta), delta(delta) {}
    cg_descent(real eta, real delta, real epsilon) : base_method<real>(epsilon), eta(eta), delta(delta) {}
    cg_descent(real eta, real delta, real epsilon, size_t max_iter) : base_method<real>(epsilon, max_iter), eta(eta), delta(delta) {}
    cg_descent(real eta, real delta, real epsilon, size_t max_iter, real working_precision)
        : base_method<real>(epsilon, max_iter, working_precision), eta(eta), delta(delta) {}

    void operator()(function::function<real>& f, line_search::base_line_search<real>& ls, la::vec<real>& x) {
        this->iter_count = 0;
        ls.clear_f_vals();

        this->tic();

        real f_curr = f(x);
        real f_prev = f_curr + 1;

        la::vec<real> gr = f.gradient(x);
        la::vec<real> gr_old;

        la::vec<real> pk = -gr;

        real q = 0.0;
        real c = 0.0;

        while (la::norm(gr) > this->epsilon && this->iter_count < this->max_iter && fabs(f_prev-f_curr)/(1+fabs(f_curr)) > this->working_precision) {
            ++this->iter_count;

            f_prev = f_curr;
            gr_old = gr;

            q *= delta;
            q += 1.0;
            c += (fabs(f_curr) - c) / q;

            ls.set_c(c);
            ls.push_f_val(f_curr);
            ls.set_current_f_val(f_curr);
            ls.set_current_g_val(gr);

            x += pk * ls(f, x, pk);

            f_curr = ls.get_current_f_val();
            gr = ls.get_current_g_val();

            // compute beta
            real eta_k = -1.0 / (la::norm(pk) * fmin(eta, la::norm(gr)));
            la::vec<real> yk = gr - gr_old;
            real py = pk.dot(yk);
            real yk_norm = la::norm(yk);
            real beta_cgd = fmax(eta_k, (1.0/py)*gr.dot(yk-pk*2*yk_norm*yk_norm/py));

            pk *= beta_cgd;
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
    real eta; // factors into the lower bound for beta
    real delta; // decay factor for q
};

}
}
}

#endif //CG_DESCENT_H_INCLUDED
