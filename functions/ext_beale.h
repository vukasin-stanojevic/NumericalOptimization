//
// Created by vukasin on 7/23/20.
//

#ifndef NUMERICALOPTIMIZATION_EXT_BEALE_H
#define NUMERICALOPTIMIZATION_EXT_BEALE_H


#include "function.h"

namespace opt {
    namespace function {

        template<class real>
        class ext_beale {
        private:
            static void calculate_f_job(const la::vec<real>* v, std::promise<real>&& prom, size_t i_start, size_t i_end) {
                real z = 0.0;
                real x_3;
                real p, d, t;
                real v_sq, v_cube;

                for (size_t i = i_start; i < i_end; i += 2) {
                    v_sq = v->get_element(i+1) * v->get_element(i+1);
                    v_cube = v->get_element(i+1) * v_sq;

                    p = 1.5 - v->get_element(i)*(1-v->get_element(i+1));
                    d = 2.25 - v->get_element(i)*(1 - v_sq);
                    t = 2.625 - v->get_element(i)*(1 - v_cube);

                    z += p*p + d*d + t*t;
                }
                prom.set_value(z);
            }

            static void calculate_grad_job(const la::vec<real>* v, la::vec<real>* grad, size_t i_start, size_t i_end) {

                real p, d, t;
                real v_sq, v_cube;

                for (size_t i = i_start; i < i_end; i += 2) {
                    v_sq = v->get_element(i+1) * v->get_element(i+1);
                    v_cube = v->get_element(i+1) * v_sq;

                    p = 1.5 - v->get_element(i)*(1-v->get_element(i+1));
                    d = 2.25 - v->get_element(i)*(1 - v_sq);
                    t = 2.625 - v->get_element(i)*(1 - v_cube);

                    (*grad)[i] = -2*p*(1 - v->get_element(i+1)) - 2*d *(1 - v_sq) - 2*t*(1 - v_cube);
                    (*grad)[i+1] = 2*p*(v->get_element(i)) + 4*d*(v->get_element(i+1))*(v->get_element(i)) + 6*t*v_sq*(v->get_element(i));
                }
            }

            static void calculate_hessian_job(const la::vec<real>* v, la::mat<real>* hess, size_t i_start, size_t i_end) {

                real p, d, t;
                real a1, a2, a3, v_sq;

                for (size_t i = i_start; i < i_end; i += 2) {
                    a1 = 1 - v->get_element(i+1);
                    a2 = 1 - v->get_element(i+1)*(v->get_element(i+1));
                    a3 = 1 - v->get_element(i+1)*(v->get_element(i+1))*(v->get_element(i+1));

                    p = 1.5 - v->get_element(i)*a1;
                    d = 2.25 - v->get_element(i)*a2;
                    t = 2.625 - v->get_element(i)*a3;
                    v_sq = v->get_element(i+1) * (v->get_element(i+1));


                    (*hess)[i][i] = 2*(a1*a1 + a2*a2 + a3*a3);
                    (*hess)[i+1][i+1] = 2*(v->get_element(i))*(v->get_element(i)) + 4*(v->get_element(i))*(v->get_element(i)) + 4*(v->get_element(i))*d - 18*(v->get_element(i))*v_sq*v_sq + 12*t*(v->get_element(i))*(v->get_element(i+1));

                    (*hess)[i+1][i] = -2*(v->get_element(i))*a1 + 2*p + 4*d*(v->get_element(i+1)) - 4*(v->get_element(i))*(v->get_element(i+1))*a2 + 6*v_sq*t - 6*v_sq*(v->get_element(i))*a3;
                    (*hess)[i][i+1] = (*hess)[i+1][i];
                }
            }
        public:
            static real func(const la::vec<real>& v) {
                if (v.size() == 0) {
                    throw "ext_beale: n must be even and positive";
                }
                return function<real>::calculate_value_multithread(&v, ext_beale<real>::calculate_f_job);
            }

            static la::vec<real> gradient(const la::vec<real>& v) {
                if (v.size() == 0) {
                    throw "ext_beale: n must be even and positive";
                }

                size_t n = v.size();
                la::vec<real> grad(n);

                function<real>::calculate_gradient_multithread(&v, &grad, ext_beale<real>::calculate_grad_job);

                return grad;
            }

            static la::mat<real> hessian(const la::vec<real>& v) {
                if (v.size() == 0) {
                    throw "ext_beale: n must be even and positive";
                }

                size_t n = v.size();
                la::mat<real> hes(n, n, 0.0);

                function<real>::calculate_hessian_multithread(&v, &hes, ext_beale<real>::calculate_hessian_job);

                return hes;
            }

            static la::vec<real> starting_point(const size_t n) {
                if (n == 0) {
                    throw "ext_beale: n must be even and positive";
                }

                la::vec<real> st(n);
                for (int i = 0; i < n; i+=2) {
                    st[i] = 1;
                    st[i+1] = 0.8;
                }

                return st;
            }

            static function<real> getFunction() {
                return function<real>(func, gradient, hessian, starting_point);
            }
        };

    }
}

#endif //NUMERICALOPTIMIZATION_EXT_BEALE_H
