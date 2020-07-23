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
        public:
            static real func(const la::vec<real>& v) {
                if (v.size() == 0) {
                    throw "hager_function: n must be even and positive";
                }

                size_t n = v.size();
                real z = 0.0;

                for (size_t i = 0; i < n; i++) {
                    z += exp(v[i]) - sqrt(i+1)*v[i];
                }

                return z;
            }

            static la::vec<real> gradient(const la::vec<real>& v) {
                if (v.size() == 0) {
                    throw "hager_function: n must be even and positive";
                }

                size_t n = v.size();
                la::vec<real> grad(n);

                real s = 0;
                real tmp2;

                for (size_t i = 0; i < n; i++) {
                    grad[i] = exp(v[i]) - sqrt(i+1);
                }

                return grad;
            }

            static la::mat<real> hessian(const la::vec<real>& v) {
                if (v.size() == 0) {
                    throw "hager_function: n must be even and positive";
                }

                size_t n = v.size();
                la::mat<real> hes(n, n, 0);

                for (size_t i = 0; i < n; i++) {
                    hes[i][i] = exp(v[i]);
                }

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
