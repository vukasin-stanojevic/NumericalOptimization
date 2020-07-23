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
        public:
            static real func(const la::vec<real>& v) {
                if (v.size() == 0) {
                    throw "ext_penalty: n must be even and positive";
                }

                size_t n = v.size();
                real z = 0.0;
                real sqr = 0;
                real tmp, tmp2;

                for (size_t i = 0; i < n; i++) {
                    tmp = v[i] - 1;
                    tmp2 = v[i]*v[i] - 0.25;
                    z += tmp;
                    sqr += tmp2;
                }

                return z + sqr*sqr;
            }

            static la::vec<real> gradient(const la::vec<real>& v) {
                if (v.size() == 0) {
                    throw "ext_penalty: n must be even and positive";
                }

                size_t n = v.size();
                la::vec<real> grad(n);

                real s = 0;
                real tmp2;

                for (size_t i = 0; i < n; i++) {
                    tmp2 = v[i]*v[i] - 0.25;
                    s += tmp2;
                }
                for (size_t i = 0; i < n; i++) {
                    grad[i] = 2 * (v[i] - 1) + 4 * v[i] * s;
                }

                return grad;
            }

            static la::mat<real> hessian(const la::vec<real>& v) {
                if (v.size() == 0) {
                    throw "ext_penalty: n must be even and positive";
                }

                size_t n = v.size();
                la::mat<real> hes(n, n);

                real s = 0;
                real tmp2;

                for (size_t i = 0; i < n; i++) {
                    tmp2 = v[i]*v[i] - 0.25;
                    s += tmp2;
                }


                for (size_t i = 0; i < n; i++)
                    for (size_t j = 0; j<=i; j++)
                    {
                        if (i == j) {
                            hes[i][j] = 2 + 4 * s + 8 * v[i];
                        } else {
                            hes[i][j] = 8 * v[i] * v[j];
                            hes[j][i] = hes[i][j];
                        }
                    }

                return hes;
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
