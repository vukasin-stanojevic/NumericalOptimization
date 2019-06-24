#ifndef NUMERICALOPTIMIZATION_EXTENDED_QP1_H
#define NUMERICALOPTIMIZATION_EXTENDED_QP1_H

#include "function.h"

namespace opt {
namespace function {

template<class real>
class ext_quad_pen_qp1 {
public:
    static real func(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "ext_quad_pen_qp1: n must be positive";
        auto n = v.size();
        real s1 = 0;
        real s2 = 0;
        for (size_t i=0; i<n-1; i++) {
            real t = v[i]*v[i] - 2;
            s1 += t*t;

            t = v[i]*v[i];
            s2+=t;
        }
        s2 += v[n-1]*v[n-1] - 0.5;

        return s1 + s2*s2;
    }

    static la::vec<real> gradient(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "ext_quad_pen_qp1: n must be positive";

        la::vec<real> z(v.size(), 0.0);

        real sum_sq = -0.5;
        for (size_t i=0; i<v.size(); ++i)
            sum_sq += v[i]*v[i];

        for (size_t i=0; i<v.size(); i++){
            z[i]+=4*sum_sq*v[i] + 4*(v[i]*v[i] - 2)*v[i];
        }
        auto n = v.size();
        z[n-1] -= 4*(v[n-1]*v[n-1] - 2)*v[n-1];

        return z;
    }

    static la::mat<real> hessian(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "ext_quad_pen_qp1: n must be positive";


        la::mat<real> z(v.size(), v.size(), 0.0);

        real sum_sq = -0.5;
        for (size_t i=0; i<v.size(); ++i)
            sum_sq += v[i]*v[i];
        auto n = v.size();
        for(int i=0;i<n;++i){
            for(int j=0;j<n;++j){
                if(i==j){
                    z[i][i] = 4*(3*v[i]*v[i] - 2) + 4*(2*v[i]*v[i] + sum_sq);
                }else{
                    z[i][j] = 8*v[i]*v[j];
                }
            }
        }
        z[n-1][n-1]-=4*(3*v[n-1]*v[n-1] - 2);

        return z;
    }

    static la::vec<real> starting_point(const size_t n) {
        if (n == 0)
            throw "ext_quad_pen_qp1: n must be positive";
        return la::vec<real>(n, 0.5);
    }

    static function<real> getFunction(){
        return function<real>(func, gradient, hessian, starting_point);
    }
};
}
}

#endif //NUMERICALOPTIMIZATION_EXTENDED_QP1_H
