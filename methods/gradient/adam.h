//
// Created by vukasin on 7/23/20.
//

#ifndef NUMERICALOPTIMIZATION_ADAM_H
#define NUMERICALOPTIMIZATION_ADAM_H


namespace opt {
    namespace method {
        namespace gradient {

            template<class real>
            class adam : public base_method<real> {
            private:
                real beta_1 = (real)0.9;
                real beta_2 = (real)0.999;
                real eps = (real)10e-8;
            public:
                explicit adam(real beta_1 = 0.9, real beta_2 = 0.999, real eps = 10e-8): base_method<real>(), beta_1(beta_1), beta_2(beta_2), eps(eps) {
                    this->method_name = "Adam";
                }
                void operator()(function::function<real>& f, line_search::base_line_search<real>& ls, la::vec<real>& x) {
                    this->iter_count = 0;
                    ls.clear_f_vals();
                    this->gr_norms.clear();

                    this->tic();

                    real f_curr = f(x);
                    real f_prev = f_curr + 1;

                    la::vec<real> gr = f.gradient(x);
                    la::vec<real> m(gr.size(), 0);
                    la::vec<real> v(gr.size(), 0);
                    la::vec<real> p(x.size());

                    real t;
                    real gr_norm = la::norm(gr);
                    this->gr_norms.push_back(gr_norm);

                    while (gr_norm > this->epsilon && this->iter_count < this->max_iter &&
                           fabs(f_prev - f_curr) / (1 + fabs(f_curr)) > this->working_precision) {
                        ++this->iter_count;

                        ls.push_f_val(f_curr);
                        ls.set_current_f_val(f_curr);
                        ls.set_current_g_val(gr);

                        m *= beta_1;
                        m += gr * (1 - beta_1);

                        v *= beta_2;
                        v += gr * gr * (1 - beta_2);

                        compute_p(&m, &v, eps, beta_1, beta_2, &p, this->iter_count);
                        t = ls(f, x, p);

                        x += p * t;
                        f_prev = f_curr;
                        f_curr = ls.get_current_f_val();
                        gr = ls.get_current_g_val();
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
                static void compute_p(la::vec<real>* m, la::vec<real>* v, real eps, real beta_1, real beta_2, la::vec<real>* p, int step) {
                    unsigned int processor_count = std::thread::hardware_concurrency();
                    processor_count = processor_count > la::MAX_THREAD_NUM ? la::MAX_THREAD_NUM : processor_count;
                    if (processor_count > 1) {
                        std::vector<std::thread> threads;
                        size_t work_by_thread = m->size() / processor_count;
                        int last_thread_additional_work = m->size() - work_by_thread * processor_count;

                        int k;
                        for (k = 0; k < processor_count - 1; k++) {
                            threads.push_back(std::thread(&adam<real>::compute_p_task, m, v, eps, beta_1, beta_2, p, step, k*work_by_thread, (k+1)*work_by_thread));
                        }
                        threads.push_back(std::thread(&adam<real>::compute_p_task, m, v, eps, beta_1, beta_2, p, step, k*work_by_thread, (k+1)*work_by_thread + last_thread_additional_work));

                        for (auto& th : threads) th.join();
                    } else {
                        compute_p_task(m, v, eps, beta_1, beta_2, p, step, 0, m->size());
                    }
                }
                static void compute_p_task(la::vec<real>* m, la::vec<real>* v, real eps, real beta_1, real beta_2, la::vec<real>* p, int step, int i_start, int i_end) {
                    real m_corr;
                    real v_corr;

                    for (int i=i_start; i<i_end; ++i) {
                        m_corr = (*m)[i] / (1 - pow(beta_1, step));
                        v_corr = (*v)[i] / (1 - pow(beta_2, step));
                        (*p)[i] = -1 * m_corr /  (sqrt(v_corr) + eps);
                    }
                }
            };
        }
    }
}


#endif //NUMERICALOPTIMIZATION_ADAM_H
