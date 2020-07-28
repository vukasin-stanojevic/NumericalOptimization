//
// Created by vukasin on 7/22/20.
//

#ifndef NUMERICALOPTIMIZATION_ADADELTA_H
#define NUMERICALOPTIMIZATION_ADADELTA_H

namespace opt {
    namespace method {
        namespace gradient {

            template<class real>
            class adadelta : public base_method<real> {
            private:
                real gamma = (real)0.9;
                real eps = (real)10e-8;
            public:
                explicit adadelta(real _gamma = 0.9, real eps = 10e-8): base_method<real>(), gamma(_gamma), eps(eps) {
                    this->method_name = "Adadelta";
                }
                void operator()(function::function<real>& f, line_search::base_line_search<real>& ls, la::vec<real>& x) {
                    this->iter_count = 0;
                    this->gr_norms.clear();
                    ls.clear_f_vals();

                    this->tic();

                    real f_curr = f(x);
                    real f_prev = f_curr + 1;

                    la::vec<real> gr = f.gradient(x);
                    la::vec<real> gr_sq(x.size(), 0);

                    la::vec<real> update_sq(x.size(), 0);
                    la::vec<real> p(x.size());

                    real gr_norm = la::norm(gr);
                    this->gr_norms.push_back(gr_norm);

                    while (gr_norm > this->epsilon && this->iter_count < this->max_iter && fabs(f_prev-f_curr)/(1+fabs(f_curr)) > this->working_precision) {
                        gr_sq *= gamma;
                        gr_sq += gr * gr * (1 - gamma);

                        compute_p(&gr_sq, &update_sq, &gr, eps, &p);
                        update_sq *= gamma;
                        update_sq += p * p * (1 - gamma);

                        x += p;

                        f_prev = f_curr;
                        f_curr = f(x);
                        gr = f.gradient(x);

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
                static void compute_p(la::vec<real>* sq_grads, la::vec<real>* sq_updates, la::vec<real>* grad, real eps, la::vec<real>* p) {
                    unsigned int processor_count = std::thread::hardware_concurrency();
                    processor_count = processor_count > la::MAX_THREAD_NUM ? la::MAX_THREAD_NUM : processor_count;
                    if (processor_count > 1) {
                        std::vector<std::thread> threads;
                        size_t work_by_thread = grad->size() / processor_count;
                        int last_thread_additional_work = grad->size() - work_by_thread * processor_count;

                        int k;
                        for (k = 0; k < processor_count - 1; k++) {
                            threads.push_back(std::thread(&adadelta<real>::compute_p_task, sq_grads, sq_updates, grad, eps, p, k*work_by_thread, (k+1)*work_by_thread));
                        }
                        threads.push_back(std::thread(&adadelta<real>::compute_p_task, sq_grads, sq_updates, grad, eps, p, k*work_by_thread, (k+1)*work_by_thread + last_thread_additional_work));

                        for (auto& th : threads) th.join();
                    } else {
                        compute_p_task(sq_grads, sq_updates, grad, eps, p, 0, grad->size());
                    }
                }
                static void compute_p_task(la::vec<real>* sq_grads, la::vec<real>* sq_updates, la::vec<real>* grad, real eps, la::vec<real>* p, int i_start, int i_end) {
                    for (int i=i_start; i<i_end; ++i) {
                        (*p)[i] = -1 * (*grad)[i] * sqrt((*sq_updates)[i] + eps) /  sqrt((*sq_grads)[i] + eps);
                    }
                }
            };
        }
    }
}

#endif //NUMERICALOPTIMIZATION_ADADELTA_H
