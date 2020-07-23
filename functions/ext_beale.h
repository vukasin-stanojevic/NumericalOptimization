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
        public:
            static real func(const la::vec<real>& v) {
                if (v.size() == 0) {
                    throw "ext_beale: n must be even and positive";
                }

                size_t n = v.size();
                real z = 0.0;
                real x_3;
                real p, d, t;

                for (size_t i = 0; i < n; i += 2) {
                    p = 1.5 - v[i]*(1-v[i+1]);
                    d = 2.25 - v[i]*(1 - v[i+1]*v[i+1]);
                    t = 2.625 - v[i]*(1 - v[i+1]*v[i+1]*v[i+1]);

                    z += p*p + d*d + t*t;
                }

                return z;
            }

            static la::vec<real> gradient(const la::vec<real>& v) {
                if (v.size() == 0) {
                    throw "ext_beale: n must be even and positive";
                }

                size_t n = v.size();
                la::vec<real> grad(n);

                real p, d, t;

                for (size_t i = 0; i < n; i += 2) {
                    p = 1.5 - v[i]*(1-v[i+1]);
                    d = 2.25 - v[i]*(1 - v[i+1]*v[i+1]);
                    t = 2.625 - v[i]*(1 - v[i+1]*v[i+1]*v[i+1]);

                    grad[i] = -2*p*(1 - v[i+1]) - 2*d *(1 -v[i+1]*v[i+1]) - 2*t*(1 - v[i+1]*v[i+1]*v[i+1]);
                    grad[i+1] = 2*p*v[i] + 4*d*v[i+1]*v[i] + 6*t*v[i+1]*v[i+1]*v[i];
                }

                return grad;
            }

            static la::mat<real> hessian(const la::vec<real>& v) {
                if (v.size() == 0) {
                    throw "ext_beale: n must be even and positive";
                }

                size_t n = v.size();
                la::mat<real> hes(n, n, 0.0);
                real p, d, t;
                real a1, a2, a3;

                for (size_t i = 0; i < n; i += 2) {
                    p = 1.5 - v[i]*(1-v[i+1]);
                    d = 2.25 - v[i]*(1 - v[i+1]*v[i+1]);
                    t = 2.625 - v[i]*(1 - v[i+1]*v[i+1]*v[i+1]);

                    a1 = 1 - v[i+1];
                    a2 = 1 - v[i+1]*v[i+1];
                    a3 = 1 - v[i+1]*v[i+1]*v[i+1];

                    hes[i][i] = 2*(a1*a1 + a2*a2 + a3*a3);
                    hes[i+1][i+1] = 2*v[i]*v[i] + 4*v[i]*v[i]*v[i+1]*v[i+1] + 4*v[i]*d - 18*v[i]*v[i+1]*v[i+1]*v[i+1]*v[i+1] + 12*t*v[i]*v[i+1];

                    hes[i+1][i] = -2*v[i]*a1 + 2*p + 4*d*v[i+1] - 4*v[i]*v[i+1]*a2 + 6*v[i+1]*v[i+1]*t - 6*v[i+1]*v[i+1]*v[i]*a3;
                    hes[i][i+1] = hes[i+1][i];
                }

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
