//
// Created by lazar on 26.5.19..
//

#ifndef PROJEKATC___EXTENDED_QP2_H
#define PROJEKATC___EXTENDED_QP2_H


#include "function.h"

namespace functions{
template<class real>
class extended_qp2{
public:
    static real func(const la::vec<real>& v){
        if (v.size() == 0)
            throw "extended_qp2: n must be positive";
        real z = 0;
        for (size_t i=0; i<v.size()-1; i++) {
            real t = v[i]*v[i] - sin(v[i]);
            z += t*t;
        }
        real p = -100;
        for (size_t i=0; i<v.size(); i++)
            p += v[i]*v[i];
        z += p*p;
        return z;
    }

    static la::vec<real> gradient(const la::vec<real>& v){
        if (v.size() == 0)
            throw "extended_qp2: n must be positive";
        la::vec<real> z(v.size(), 0.0);
        for (size_t i=0; i<v.size()-1; i++)
            z[i] += 2*(2*v[i]-cos(v[i]))*(v[i]*v[i] - sin(v[i]));
        real t = -100;
        for (size_t i=0; i<v.size(); i++)
            t += v[i]*v[i];
        for (size_t i=0; i<v.size(); i++)
            z[i] += 4*v[i]*t;
        return z;
    }

    static la::mat<real> hessian(const la::vec<real>& v){
        if (v.size() == 0)
            throw "extended_qp2: n must be positive";
        la::mat<real> z(v.size(), v.size(), 0.0);

        real sk = 0;
        for (size_t i=0; i<v.size(); i++)
            sk += v[i]*v[i];

        for (size_t i=0; i<v.size(); i++)
            for (size_t j=i+1; j<v.size(); j++)
                z[i][j] = z[j][i] = v[i]*v[j]*8;

        z[v.size()-1][v.size()-1] = 8*v[v.size()-1]*v[v.size()-1]
                                    + 4*(sk - 100);

        for (size_t i=0; i<v.size()-1; i++) {
            real a = v[i];
            z[i][i] = 8*a*a + 4*(sk - 100) + 2*(2*a - cos(a))*(2*a - cos(a))
                      + 2*(a*a-sin(a))*(2+sin(a));
        }

        return z;
    }

    static la::vec<real> starting_point(const size_t n){
        if (n == 0)
            throw "extended_qp2: n must be positive";
        return la::vec<real>(n, 0.5);
    }
    static function<real> getFunction(){
        return function<real>(func,gradient,hessian,starting_point);
    }
};
}

#endif //PROJEKATC___EXTENDED_QP2_H
