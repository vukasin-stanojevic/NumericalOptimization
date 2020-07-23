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
        public:
            static real func(const la::vec<real>& v) {
                if (v.size() == 0) {
                    throw "Extended Three exponential terms: n must be even and positive";
                }

                size_t n = v.size();
                real z = 0.0;
                real p, d, t;

                for (size_t i = 0; i < n; i += 2) {
                    p = exp(v[i] + 3 * v[i+1] - 0.1);
                    d = exp(v[i] - 3 * v[i+1] - 0.1);
                    t = exp(-v[i] - 0.1);

                    z += p + d + t;
                }

                return z;
            }

            static la::vec<real> gradient(const la::vec<real>& v) {
                if (v.size() == 0) {
                    throw "Extended Three exponential terms: n must be even and positive";
                }

                size_t n = v.size();
                la::vec<real> grad(n);

                real p, d, t;

                for (size_t i = 0; i < n; i += 2) {
                    p = exp(v[i] + 3 * v[i+1] - 0.1);
                    d = exp(v[i] - 3 * v[i+1] - 0.1);
                    t = exp(-v[i] - 0.1);

                    grad[i] = p + d - t;
                    grad[i+1] = 3*(p-d);
                }

                return grad;
            }

            static la::mat<real> hessian(const la::vec<real>& v) {
                if (v.size() == 0) {
                    throw "Extended Three exponential terms: n must be even and positive";
                }

                size_t n = v.size();
                la::mat<real> hes(n, n, 0.0);
                real p, d, t;

                for (size_t i = 0; i < n; i += 2) {
                    p = exp(v[i] + 3 * v[i+1] - 0.1);
                    d = exp(v[i] - 3 * v[i+1] - 0.1);
                    t = exp(-v[i] - 0.1);

                    hes[i][i] = p + d + t;
                    hes[i+1][i] = 3 * (p - d);
                    hes[i][i+1] = hes[i+1][i];
                    hes[i+1][i+1] = 3 * (p+d);
                }

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
