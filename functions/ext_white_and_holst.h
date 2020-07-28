//
// Created by vukasin on 7/23/20.
//

#ifndef NUMERICALOPTIMIZATION_EXT_WHITE_AND_HOLST_H
#define NUMERICALOPTIMIZATION_EXT_WHITE_AND_HOLST_H


#include "function.h"

namespace opt {
    namespace function {

        template<class real>
        class ext_white_and_holst {
        private:
            static void calculate_f_job(const la::vec<real>* v, std::promise<real>&& prom, size_t i_start, size_t i_end) {
                real z = 0.0;
                real x_3;
                real tmp;

                for (size_t i = i_start; i < i_end; i += 2) {
                    x_3 = (*v)[i] * (*v)[i] * (*v)[i];
                    tmp = (*v)[i+1] - x_3;
                    z += 100 * tmp * tmp + (1 - (*v)[i]) * (1 - (*v)[i]);
                }
                prom.set_value(z);
            }

            static void calculate_grad_job(const la::vec<real>* v, la::vec<real>* grad, size_t i_start, size_t i_end) {
                real x_3;
                real tmp;

                for (size_t i = i_start; i < i_end; i += 2) {
                    x_3 = (*v)[i] * (*v)[i] * (*v)[i];
                    tmp = (*v)[i+1] - x_3;
                    (*grad)[i] = -2*100*tmp*3*(*v)[i]*(*v)[i] - 2 * (1 - (*v)[i]);
                    (*grad)[i+1] = 2*100*tmp;
                }
            }

            static void calculate_hessian_job(const la::vec<real>* v, la::mat<real>* hess, size_t i_start, size_t i_end) {

                for (size_t i = i_start; i < i_end; i += 2) {
                    (*hess)[i][i] = -12*100*(*v)[i+1]*(*v)[i] + 30*100*(*v)[i]*(*v)[i]*(*v)[i]*(*v)[i];
                    (*hess)[i+1][i] = -6*100*(*v)[i]*(*v)[i];
                    (*hess)[i][i+1] = (*hess)[i+1][i];
                    (*hess)[i+1][i+1] = 2*100;
                }
            }
        public:
            static real func(const la::vec<real>& v) {
                if (v.size() == 0) {
                    throw "ext_white_and_holst: n must be even and positive";
                }
                return function<real>::calculate_value_multithread(&v, ext_white_and_holst<real>::calculate_f_job);
            }

            static la::vec<real> gradient(const la::vec<real>& v) {
                if (v.size() == 0) {
                    throw "ext_white_and_holst: n must be even and positive";
                }

                size_t n = v.size();
                la::vec<real> grad(n);

                function<real>::calculate_gradient_multithread(&v, &grad, ext_white_and_holst<real>::calculate_grad_job);

                return grad;
            }

            static la::mat<real> hessian(const la::vec<real>& v) {
                if (v.size() == 0) {
                    throw "ext_white_and_holst: n must be even and positive";
                }

                size_t n = v.size();
                la::mat<real> hes(n, n, 0.0);

                function<real>::calculate_hessian_multithread(&v, &hes, ext_white_and_holst<real>::calculate_hessian_job);

                return hes;
            }

            static la::vec<real> starting_point(const size_t n) {
                if (n == 0) {
                    throw "ext_white_and_holst: n must be even and positive";
                }

                la::vec<real> st(n);
                for (int i = 0; i < n; i+=2) {
                    st[i] = -1.2;
                    st[i+1] = 1;
                }

                return st;
            }

            static function<real> getFunction() {
                return function<real>(func, gradient, hessian, starting_point);
            }
        };

    }
}

#endif //NUMERICALOPTIMIZATION_EXT_WHITE_AND_HOLST_H
