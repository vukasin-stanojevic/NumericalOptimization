//
// Created by vukasin on 7/23/20.
//

#ifndef NUMERICALOPTIMIZATION_HAGER_H
#define NUMERICALOPTIMIZATION_HAGER_H

#include "function.h"

namespace opt {
    namespace function {

        template<class real>
        class hager_function {
        private:
            static void calculate_f_job(const la::vec<real>* v, std::promise<real>&& prom, size_t i_start, size_t i_end) {
                real z = 0.0;
                for (size_t i = i_start; i < i_end; i++) {
                    z += exp((*v)[i]) - sqrt(i+1)*(*v)[i];
                }

                prom.set_value(z);
            }

            static void calculate_grad_job(const la::vec<real>* v, la::vec<real>* grad, size_t i_start, size_t i_end) {
                for (size_t i = i_start; i < i_end; i++) {
                    (*grad)[i] = exp((*v)[i]) - sqrt(i+1);
                }
            }

            static void calculate_hessian_job(const la::vec<real>* v, la::mat<real>* hess, size_t i_start, size_t i_end) {
                for (size_t i = i_start; i < i_end; i++) {
                    (*hess)[i][i] = exp((*v)[i]);
                }
            }
        public:
            static real func(const la::vec<real>& v) {
                if (v.size() == 0) {
                    throw "hager_function: n must be even and positive";
                }
                return function<real>::calculate_value_multithread(&v, hager_function<real>::calculate_f_job);
            }

            static la::vec<real> gradient(const la::vec<real>& v) {
                if (v.size() == 0) {
                    throw "hager_function: n must be even and positive";
                }

                size_t n = v.size();
                la::vec<real> grad(n);

                function<real>::calculate_gradient_multithread(&v, &grad, hager_function<real>::calculate_grad_job);

                return grad;
            }

            static la::mat<real> hessian(const la::vec<real>& v) {
                if (v.size() == 0) {
                    throw "hager_function: n must be even and positive";
                }

                size_t n = v.size();
                la::mat<real> hes(n, n, 0.0);

                function<real>::calculate_hessian_multithread(&v, &hes, hager_function<real>::calculate_hessian_job);

                return hes;
            }

            static la::vec<real> starting_point(const size_t n) {
                if (n == 0) {
                    throw "hager_function: n must be even and positive";
                }

                la::vec<real> st(n, 1);

                return st;
            }

            static function<real> getFunction() {
                return function<real>(func, gradient, hessian, starting_point);
            }
        };

    }
}

#endif //NUMERICALOPTIMIZATION_HAGER_H
