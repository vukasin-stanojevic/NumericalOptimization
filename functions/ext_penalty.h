//
// Created by vukasin on 7/23/20.
//

#ifndef NUMERICALOPTIMIZATION_EXT_PENALTY_H
#define NUMERICALOPTIMIZATION_EXT_PENALTY_H

#include "function.h"

namespace opt {
    namespace function {

        template<class real>
        class ext_penalty {
        private:

            static void calculate_hess_job(const la::vec<real>& v, real s, la::mat<real>& hess, size_t i_start, size_t i_end) {
                for (size_t i = i_start; i < i_end; i++)
                    for (size_t j = 0; j<=i; j++)
                    {
                        if (i == j) {
                            hess[i][j] = 2 + 4 * s + 8 * v[i];
                        } else {
                            hess[i][j] = 8 * v[i] * v[j];
                            hess[j][i] = hess[i][j];
                        }
                    }
            }
        public:
            static real func(const la::vec<real>& v) {
                if (v.size() == 0) {
                    throw "ext_penalty: n must be even and positive";
                }
                real z = 0.0;
                real sqr = 0;
                real tmp, tmp2;

                for (size_t i = 0; i < v.size() - 1; i++) {
                    tmp = v[i] - 1;
                    z += tmp*tmp;
                    sqr += v[i]*v[i];
                }
                sqr += v[v.size()-1] * v[v.size() - 1];
                sqr -= 0.25;
                sqr *= sqr;

                return z + sqr;
            }

            static la::vec<real> gradient(const la::vec<real>& v) {
                if (v.size() == 0) {
                    throw "ext_penalty: n must be even and positive";
                }
                la::vec<real> grad(v.size());

                real s = 0.0;
                real sqr = 0;
                real tmp, tmp2;

                for (size_t i = 0; i < v.size() - 1; i++) {
                    s += v[i] - 1;
                    sqr += v[i]*v[i];
                }
                sqr += v[v.size()-1] * v[v.size() - 1];
                sqr -= 0.25;


                for (size_t i = 0; i < v.size() - 1; i++) {
                    grad[i] = 2 * s + 4 * sqr * v[i];
                }
                grad[v.size()-1] = 4 * sqr * v[v.size() - 1];

                return grad;
            }

            static la::mat<real> hessian(const la::vec<real>& v) {
                if (v.size() == 0) {
                    throw "ext_penalty: n must be even and positive";
                }

                size_t n = v.size();
                la::mat<real> hess(n, n);

                la::vec<real> tmp(n, 1);
                real s = tmp.dot(v*v) - 0.25;

                unsigned int processor_count = std::thread::hardware_concurrency();
                processor_count = processor_count > la::MAX_THREAD_NUM ? la::MAX_THREAD_NUM : processor_count;
                if (processor_count % 2 == 1) {
                    processor_count--;
                }

                if (processor_count > 1) {
                    std::vector<std::thread> threads;

                    size_t work_by_thread = hess.rows() / processor_count;
                    int last_thread_additional_work = hess.rows() - work_by_thread * processor_count;

                    int k;
                    for (k = 0; k < processor_count - 1; k++) {
                        threads.push_back(std::thread(&ext_penalty<real>::calculate_hess_job, std::ref(v), s, std::ref(hess), k*work_by_thread, (k+1)*work_by_thread));
                    }
                    threads.push_back(std::thread(&ext_penalty<real>::calculate_hess_job, std::ref(v), s, std::ref(hess), k*work_by_thread, (k+1)*work_by_thread + last_thread_additional_work));

                    for (int i = 0; i < threads.size(); ++i) {
                        threads[i].join();
                    }
                } else {
                    ext_penalty<real>::calculate_hess_job(v, s, hess, 0, hess.rows());
                }

                return hess;
            }

            static la::vec<real> starting_point(const size_t n) {
                if (n == 0) {
                    throw "ext_penalty: n must be even and positive";
                }

                la::vec<real> st(n);
                for (int i = 0; i < n; i++) {
                    st[i] = i+1;
                }

                return st;
            }

            static function<real> getFunction() {
                return function<real>(func, gradient, hessian, starting_point);
            }
        };

    }
}

#endif //NUMERICALOPTIMIZATION_EXT_PENALTY_H
