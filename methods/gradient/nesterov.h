//
// Created by vukasin on 7/21/20.
//

#ifndef NUMERICALOPTIMIZATION_NESTEROV_H
#define NUMERICALOPTIMIZATION_NESTEROV_H
#include "../base_method.h"

namespace opt {
    namespace method {
        namespace gradient {

            template<class real>
            class nesterov : public base_method<real> {
            private:
                real gamma = (real)0.9;
            public:
                explicit nesterov(real _gamma = 0.9): base_method<real>(), gamma(_gamma){
                    this->method_name = "Nesterov momentum";
                }
                void operator()(function::function<real>& f, line_search::base_line_search<real>& ls, la::vec<real>& x) {
                    this->iter_count = 0;
                    ls.clear_f_vals();
                    this->gr_norms.clear();

                    this->tic();

                    real f_curr = f(x);
                    real f_prev = f_curr + 1;

                    la::vec<real> gr = f.gradient(x);
                    la::vec<real> p_prev;
                    real t;

                    real gr_norm = la::norm(gr);
                    while (gr_norm > this->epsilon && this->iter_count < this->max_iter && fabs(f_prev-f_curr)/(1+fabs(f_curr)) > this->working_precision) {
                        ls.push_f_val(f_curr);
                        ls.set_current_f_val(f_curr);
                        ls.set_current_g_val(f.gradient(x));

                        if (this->iter_count == 0) {
                            p_prev = -gr;
                            t = ls(f,x,p_prev);
                            //t = 0.0001;
                            p_prev *= t;

                        } else {
                            gr = f.gradient(x - p_prev * gamma);

                            la::vec<real> tmp = -(x - p_prev * gamma);
                            la::vec<real> tmp2 = -gr;

                            //t = ls(f, tmp, tmp2);
                            t = 0.0001;
                            p_prev *= gamma;
                            p_prev += (gr*t);
                        }

                        ++this->iter_count;
                        f_prev = f_curr;
                        x -= p_prev;
                        f_curr = f(x);
                        gr_norm = la::norm(f.gradient(x));
                        this->gr_norms.push_back(gr_norm);
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



#endif //NUMERICALOPTIMIZATION_NESTEROV_H
