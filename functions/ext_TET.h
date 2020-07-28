//
// Created by vukasin on 7/24/20.
//

#ifndef NUMERICALOPTIMIZATION_EXT_TET_H
#define NUMERICALOPTIMIZATION_EXT_TET_H


#include "function.h"

namespace opt {
    namespace function {

        template<class real>
        class ext_TET {
        private:
            static void calculate_f_job(const la::vec<real>* v, std::promise<real>&& prom, size_t i_start, size_t i_end) {
                real z = 0.0;
                real p, d, t;

                for (size_t i = i_start; i < i_end; i += 2) {
                    p = exp(v->get_element(i) + 3 * v->get_element(i + 1) - 0.1);
                    d = exp(v->get_element(i) - 3 * v->get_element(i + 1) - 0.1);
                    t = exp(-(v->get_element(i)) - 0.1);

                    z += p + d + t;
                }
                prom.set_value(z);
            }

            static void calculate_grad_job(const la::vec<real>* v, la::vec<real>* grad, size_t i_start, size_t i_end) {

                real p, d, t;

                for (size_t i = i_start; i < i_end; i += 2) {
                    p = exp(v->get_element(i) + 3 * v->get_element(i+1) - 0.1);
                    d = exp(v->get_element(i) - 3 * v->get_element(i+1) - 0.1);
                    t = exp(-(v->get_element(i)) - 0.1);

                    grad->set_element(i, p + d - t);
                    grad->set_element(i+1, 3*(p-d));
                }
            }

            static void calculate_hessian_job(const la::vec<real>* v, la::mat<real>* hess, size_t i_start, size_t i_end) {

                real p, d, t;

                for (size_t i = i_start; i < i_end; i += 2) {
                    p = exp(v->get_element(i) + 3 * v->get_element(i+1) - 0.1);
                    d = exp(v->get_element(i) - 3 * v->get_element(i+1) - 0.1);
                    t = exp(-(v->get_element(i)) - 0.1);

                    hess->set_element(i, i, p + d + t);
                    hess->set_element(i+1, i, 3 * (p - d));
                    hess->set_element(i, i+1, hess->get_element(i+1, i));
                    hess->set_element(i+1, i+1, 3 * (p+d));
                }
            }
        public:
            static real func(const la::vec<real>& v) {
                if (v.size() == 0) {
                    throw "Extended Three exponential terms: n must be even and positive";
                }
                return function<real>::calculate_value_multithread(&v, ext_TET<real>::calculate_f_job);
            }

            static la::vec<real> gradient(const la::vec<real>& v) {
                if (v.size() == 0) {
                    throw "Extended Three exponential terms: n must be even and positive";
                }
                size_t n = v.size();
                la::vec<real> grad(n);

                function<real>::calculate_gradient_multithread(&v, &grad, ext_TET<real>::calculate_grad_job);

                return grad;
            }

            static la::mat<real> hessian(const la::vec<real>& v) {
                if (v.size() == 0) {
                    throw "Extended Three exponential terms: n must be even and positive";
                }

                size_t n = v.size();
                la::mat<real> hes(n, n, 0.0);
                function<real>::calculate_hessian_multithread(&v, &hes, ext_TET<real>::calculate_hessian_job);

                return hes;
            }

            static la::vec<real> starting_point(const size_t n) {
                if (n == 0) {
                    throw "Extended Three exponential terms: n must be even and positive";
                }

                la::vec<real> st(n, 0.1);

                return st;
            }

            static function<real> getFunction() {
                return function<real>(func, gradient, hessian, starting_point);
            }
        };

    }
}

#endif //NUMERICALOPTIMIZATION_EXT_TET_H
