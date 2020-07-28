#ifndef NUMERICALOPTIMIZATION_GENERALIZED_PSC1_H
#define NUMERICALOPTIMIZATION_GENERALIZED_PSC1_H

#include "function.h"

namespace opt {
namespace function {

template<class real>
class gen_psc1 {
private:
    static void calculate_f_job(const la::vec<real>* v, std::promise<real>&& prom, size_t i_start, size_t i_end) {
        i_end = i_end == v->size()? i_end - 1 : i_end;

        real z = 0;
        for (size_t i=i_start; i<i_end; ++i) {
            real t = (*v)[i]*(*v)[i] +(*v)[i+1]*(*v)[i+1] + (*v)[i]*(*v)[i+1];
            z += t*t;
            t = sin((*v)[i]);
            z += t*t;
            t = cos((*v)[i+1]);
            z += t*t;
        }

        prom.set_value(z);
    }

    static void calculate_grad_job(const la::vec<real>* v, la::vec<real>* grad, size_t i_start, size_t i_end) {
        i_end = i_end == v->size()? i_end - 1 : i_end;
        i_start = i_start == 0? 1 : i_start;

        for (size_t i=i_start; i<i_end; ++i) {
            real t1 = (*v)[i]*(*v)[i] + (*v)[i+1]*(*v)[i+1] + (*v)[i]*(*v)[i+1];
            real t2 = (*v)[i-1]*(*v)[i-1] + (*v)[i]*(*v)[i] + (*v)[i-1]*(*v)[i];
            (*grad)[i] += 2*t1*(2*(*v)[i] + (*v)[i+1]) + 2*t2*(2*(*v)[i] + (*v)[i-1]);
        }
    }

    static void calculate_hessian_job(const la::vec<real>* v, la::mat<real>* hess, size_t i_start, size_t i_end) {
        i_end = i_end == v->size()? i_end - 1 : i_end;

        for (size_t i=i_start; i<i_end; ++i) {
            // d^2 vi
            if (i == 0){
                (*hess)[i][i] = 2*(2*(*v)[0]+(*v)[1])*(2*(*v)[0]+(*v)[1]) + 4*((*v)[0]*(*v)[0]+(*v)[0]*(*v)[1]+(*v)[1]*(*v)[1]) + 2*cos((*v)[0])*cos((*v)[0]) - 2*sin((*v)[0])*sin((*v)[0]);
            }else{
                (*hess)[i][i] = 2*((*v)[i-1]+2*(*v)[i])*((*v)[i-1]+2*(*v)[i]) + 4*((*v)[i-1]*(*v)[i-1]+(*v)[i-1]*(*v)[i]+(*v)[i]*(*v)[i]) + 2*(2*(*v)[i]+(*v)[i+1])*(2*(*v)[i]+(*v)[i+1]) + 4*((*v)[i]*(*v)[i]+(*v)[i]*(*v)[i+1]+(*v)[i+1]*(*v)[i+1]);
            }

            if(i+1<v->size()){
                // d vivi+1
                (*hess)[i+1][i] = 2*( (2*(*v)[i] + (*v)[i+1])*(2*(*v)[i+1] + (*v)[i]) + ((*v)[i]*(*v)[i] + (*v)[i+1]*(*v)[i+1] + (*v)[i]*(*v)[i+1]) );
                // d vi+1vi
                (*hess)[i][i+1] = 2*( (2*(*v)[i] + (*v)[i+1])*(2*(*v)[i+1] + (*v)[i]) + ((*v)[i]*(*v)[i] + (*v)[i+1]*(*v)[i+1] + (*v)[i]*(*v)[i+1]) );
            }
        }
    }
public:
    static real func(const la::vec<real>& v) {
        if (v.size() < 2)
            throw "gen_psc1: n must be greater than 1";

        return function<real>::calculate_value_multithread(&v, gen_psc1<real>::calculate_f_job);
    }

    static la::vec<real> gradient(const la::vec<real>& v) {
        if (v.size() < 2)
            throw "gen_psc1: n must be greater than 1";
        la::vec<real> z(v.size(), 0.0);
        auto n = v.size();
        z[0]= 2* (2*v[0]+v[1]) * (v[0]*v[0] +v[0]*v[1]+v[1]*v[1]) + 2*cos(v[0])*sin(v[0]);
        z[n-1] = 2*(v[n-2]+2*v[n-1])*(v[n-2]*v[n-2]+v[n-2]*v[n-1]+v[n-1]*v[n-1])-2*cos(v[n-1])*sin(v[n-1]);;
        function<real>::calculate_gradient_multithread(&v, &z, gen_psc1<real>::calculate_grad_job);

        return z;
    }

    static la::mat<real> hessian(const la::vec<real>& v) {
        if (v.size() < 2)
            throw "gen_psc1: n must be greater than 1";
        la::mat<real> z(v.size(), v.size(), 0.0);
        size_t n = v.size();

        function<real>::calculate_hessian_multithread(&v, &z, gen_psc1<real>::calculate_hessian_job);


        z[n-1][n-1] = 2*(v[n-2]+2*v[n-1])*(v[n-2]+2*v[n-1]) + 4*(v[n-2]*v[n-2]+v[n-2]*v[n-1]+v[n-1]*v[n-1]) - 2*cos(v[n-1])*cos(v[n-1]) + 2*sin(v[n-1])*sin(v[n-1]);
        return z;
    }

    static la::vec<real> starting_point(const size_t n) {
        if (n < 2)
            throw "gen_psc1: n must be greater than 1";
        la::vec<real> z(n, 0);
        for (size_t i=0; i<n; ++i) {
            if (i % 2) {
                z[i] = 0.1;
            } else {
                z[i] = 3;
            }
        }
        return z;
    }

    static function<real> getFunction() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};

}
}

#endif //NUMERICALOPTIMIZATION_GENERALIZED_PSC1_H
