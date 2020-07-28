//
// Created by vukasin on 7/23/20.
//

#ifndef NUMERICALOPTIMIZATION_ADAMAX_H
#define NUMERICALOPTIMIZATION_ADAMAX_H


namespace opt {
    namespace method {
        namespace gradient {

            template<class real>
            class adamax : public base_method<real> {
            private:
                real beta_1 = (real)0.9;
                real beta_2 = (real)0.999;
            public:
                explicit adamax(real beta_1 = 0.9, real beta_2 = 0.999): base_method<real>(), beta_1(beta_1), beta_2(beta_2) {
                    this->method_name = "Adamax";
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
                    la::vec<real> u(gr.size(), 0);
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
                        compute_p_and_update_u(&m, &u, &gr, beta_1, beta_2, &p, this->iter_count);
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
                static void compute_p_and_update_u(la::vec<real>* m, la::vec<real>* u, la::vec<real>* gr, real beta_1, real beta_2, la::vec<real>* p, int step) {
                    unsigned int processor_count = std::thread::hardware_concurrency();
                    processor_count = processor_count > la::MAX_THREAD_NUM ? la::MAX_THREAD_NUM : processor_count;
                    if (processor_count > 1) {
                        std::vector<std::thread> threads;
                        size_t work_by_thread = m->size() / processor_count;
                        int last_thread_additional_work = m->size() - work_by_thread * processor_count;

                        int k;
                        for (k = 0; k < processor_count - 1; k++) {
                            threads.push_back(std::thread(&adamax<real>::compute_p_and_update_u_task, m, u, gr, beta_1, beta_2, p, step, k*work_by_thread, (k+1)*work_by_thread));
                        }
                        threads.push_back(std::thread(&adamax<real>::compute_p_and_update_u_task, m, u, gr, beta_1, beta_2, p, step, k*work_by_thread, (k+1)*work_by_thread + last_thread_additional_work));

                        for (auto& th : threads) th.join();
                    } else {
                        compute_p_and_update_u_task(m, u, gr, beta_1, beta_2, p, step, 0, m->size());
                    }
                }
                static void compute_p_and_update_u_task(la::vec<real>* m, la::vec<real>* u,  la::vec<real>* grad, real beta_1, real beta_2, la::vec<real>* p, int step, int i_start, int i_end) {
                    real m_corr;
                    real v_corr;
                    real u_tmp;


                    for (int i=i_start; i<i_end; ++i) {
                        u_tmp = beta_2 * (*u)[i];
                        (*u)[i] = abs((*grad)[i]) > u_tmp?  abs((*grad)[i]) : u_tmp;

                        m_corr = (*m)[i] / (1 - pow(beta_1, step));
                        (*p)[i] = -1 * m_corr /  (*u)[i];
                    }
                }
            };
        }
    }
}



#endif //NUMERICALOPTIMIZATION_ADAMAX_H
