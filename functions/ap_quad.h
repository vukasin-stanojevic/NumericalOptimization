//
// Created by lazar on 2.6.19..
//

#ifndef NUMERICALOPTIMIZATION_AP_QUAD_H
#define NUMERICALOPTIMIZATION_AP_QUAD_H

#include "function.h"

namespace functions{
    template<class real>
    class ap_quad{
    public:
        static real func(const la::vec<real>& v){
            if (v.size() == 0)
                throw "ap_quad: n must be positive";
            real z = 0;
            auto n = v.size();
            for (size_t i=0; i<v.size(); ++i) {
                real t = (i+1)*v[i]*v[i];
                z += t;
            }
            z+=(v[0]+v[n-1])*(v[0]+v[n-1])/100;
            return z;
        }

        static la::vec<real> gradient(const la::vec<real>& v)  {
            if (v.size() == 0)
                throw "ap_quad: n must be positive";
            auto n = v.size();
            la::vec<real> z(v.size(), 0.0);
            for (size_t i=0; i<n; ++i) {
                z[i] = 2*(i+1)*v[i];
            }
            z[0]+= (v[0]+v[n-1])/50.0;
            z[n-1]+= (v[0]+v[n-1])/50.0;
            return z;
        }

        static la::mat<real> hessian(const la::vec<real>& v)  {
            if (v.size() == 0)
                throw "ap_quad: n must be positive";
            la::mat<real> z(v.size(), v.size(), 0.0);
            auto n = v.size();
            for (size_t i=0; i<v.size(); ++i) {
                z[i][i] = 2*(i+1);
            }
            z[n-1][n-1] += 1/50.0;
            z[0][0] += 1/50.0;
            z[n-1][0] += 1/50.0;
            z[0][n-1] += 1/50.0;

            return z;
        }

        static la::vec<real> starting_point(const size_t n) {
            if (n == 0)
                throw "ap_quad: n must be positive";
            return la::vec<real>(n,0.5);
        }

        static function<real> getFunction(){
            return function<real>(func,gradient,hessian,starting_point);
        }
    };
}


#endif //NUMERICALOPTIMIZATION_AP_QUAD_H
