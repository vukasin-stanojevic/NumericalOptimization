//
// Created by vukasin on 7/22/20.
//

#ifndef NUMERICALOPTIMIZATION_ADAGRAD_H
#define NUMERICALOPTIMIZATION_ADAGRAD_H


namespace opt {
    namespace method {
        namespace gradient {

            template<class real>
            class adagrad : public base_method<real> {
            private:
                real eps = (real) 10e-8;
                la::vec<real> square_grads;
            public:
                explicit adagrad(real eps = 10e-8) : base_method<real>(), eps(eps) {
                    this->method_name = "Adagrad";
                }

                void
                operator()(function::function <real> &f, line_search::base_line_search <real> &ls, la::vec<real> &x) {
                    this->iter_count = 0;
                    this->gr_norms.clear();

                    ls.clear_f_vals();

                    this->tic();

                    real f_curr = f(x);
                    real f_prev = f_curr + 1;

                    la::vec<real> gr = f.gradient(x);
                    la::vec<real> p;
                    la::vec<real> tmp(x.size());
                    real t;
                    real gr_norm = la::norm(gr);
                    this->gr_norms.push_back(gr_norm);


                    while (gr_norm > this->epsilon && this->iter_count < this->max_iter &&
                           fabs(f_prev - f_curr) / (1 + fabs(f_curr)) > this->working_precision) {
                        ls.push_f_val(f_curr);
                        ls.set_current_f_val(f_curr);
                        ls.set_current_g_val(gr);

                        if (this->iter_count == 0) {
                            square_grads = gr * gr;
                        } else {
                            square_grads += gr * gr;
                        }

                        la::vec<real>::template launch_binary_op_multithread<opt::function::reciprocal_of_sqrt_of_sum<real>>(
                                &square_grads, eps, &tmp);
                        p = (-gr) * tmp;
                        t = ls(f, x, p);

                        x += p * t;
                        f_prev = f_curr;
                        f_curr = ls.get_current_f_val();
                        gr = ls.get_current_g_val();

                        ++this->iter_count;
                        gr_norm = la::norm(gr);
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

#endif //NUMERICALOPTIMIZATION_ADAGRAD_H
