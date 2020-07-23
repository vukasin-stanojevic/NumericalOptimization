//
// Created by vukasin on 7/23/20.
//

#ifndef NUMERICALOPTIMIZATION_RMSPROP_H
#define NUMERICALOPTIMIZATION_RMSPROP_H

namespace opt {
    namespace method {
        namespace gradient {

            template<class real>
            class rms_prop : public base_method<real> {
            private:
                real gamma = (real)0.9;
                real eps = (real)10e-8;
            public:
                explicit rms_prop(real _gamma = 0.9, real eps = 10e-8): base_method<real>(), gamma(_gamma), eps(eps) {}
                void operator()(function::function<real>& f, line_search::base_line_search<real>& ls, la::vec<real>& x) {
                    this->iter_count = 0;
                    ls.clear_f_vals();
                    this->gr_norms.clear();
                    this->tic();

                    real f_curr = f(x);
                    real f_prev = f_curr + 1;

                    la::vec<real> gr = f.gradient(x);
                    la::vec<real> gr_sq(x.size(), 0);
                    la::vec<real> p(x.size());
                    //la::vec<real> tmp(x.size());
                    real t;

                    real gr_norm = la::norm(gr);
                    while (gr_norm > this->epsilon && this->iter_count < this->max_iter &&
                           fabs(f_prev - f_curr) / (1 + fabs(f_curr)) > this->working_precision) {
                        ls.push_f_val(f_curr);
                        ls.set_current_f_val(f_curr);
                        ls.set_current_g_val(gr);

                        if (this->iter_count == 0) {
                            gr_sq = gr * gr;
                        } else {
                            gr_sq = gr_sq * gamma + gr * gr * (1 - gamma);
                        }

//                        la::vec<real>::template launch_binary_op_multithread<opt::function::reciprocal_of_sqrt_of_sum<real>>(
//                                &gr_sq, eps, &tmp);
//                        p = (-gr) * tmp;
                        compute_p(&gr_sq, &gr, eps, &p); // efikasniji nacin izracunavanja. daje isti rezultat kao i prethodne linije
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

            private:
                static void compute_p(la::vec<real>* sq_grads, la::vec<real>* grad, real eps, la::vec<real>* p) {
                    unsigned int processor_count = std::thread::hardware_concurrency();
                    processor_count = processor_count > MAX_THREAD_NUM ? MAX_THREAD_NUM : processor_count;
                    if (processor_count > 1) {
                        std::vector<std::thread> threads;
                        size_t work_by_thread = grad->size() / processor_count;
                        int last_thread_additional_work = grad->size() - work_by_thread * processor_count;

                        int k;
                        for (k = 0; k < processor_count - 1; k++) {
                            threads.push_back(std::thread(&rms_prop<real>::compute_p_task, sq_grads, grad, eps, p, k*work_by_thread, (k+1)*work_by_thread));
                        }
                        threads.push_back(std::thread(&rms_prop<real>::compute_p_task, sq_grads, grad, eps, p, k*work_by_thread, (k+1)*work_by_thread + last_thread_additional_work));

                        for (auto& th : threads) th.join();
                    } else {
                        compute_p_task(sq_grads, grad, eps, p, 0, grad->size());
                    }
                }
                static void compute_p_task(la::vec<real>* sq_grads, la::vec<real>* grad, real eps, la::vec<real>* p, int i_start, int i_end) {
                    for (int i=i_start; i<i_end; ++i) {
                        (*p)[i] = -1 * (*grad)[i] /  sqrt((*sq_grads)[i] + eps);
                    }
                }
            };
        }
    }
}

#endif //NUMERICALOPTIMIZATION_RMSPROP_H
