#ifndef NUMERICALOPTIMIZATION_FLETCHCR_H
#define NUMERICALOPTIMIZATION_FLETCHCR_H

#include "function.h"

namespace opt {
namespace function {

template<class real>
class fletchcr {
    static const int c = 100;
private:
    static void calculate_f_job(const la::vec<real>* v, std::promise<real>&& prom, size_t i_start, size_t i_end) {
        real z = 0;
        real t;
        i_end = i_end == v->size()? i_end - 1 : i_end;

        for (size_t i=i_start; i<i_end; i++) {
            t = ((*v)[i+1] - (*v)[i] + 1 - (*v)[i]*(*v)[i]);
            z+= c*t*t;
        }

        prom.set_value(z);
    }

    static void calculate_grad_job(const la::vec<real>* v, la::vec<real>* grad, size_t i_start, size_t i_end) {
        i_end = i_end == v->size()? i_end - 1 : i_end;
        i_start = i_start == 0? 1 : i_start;

        for(int i=i_start;i<i_end;++i){
            (*grad)[i] = -2*c*(2*(*v)[i]+1)*((*v)[i+1] - (*v)[i] + 1 - (*v)[i]*(*v)[i]) + 2*c*((*v)[i] - (*v)[i-1] + 1 - (*v)[i-1]*(*v)[i-1]);
        }
    }

    static void calculate_hessian_job(const la::vec<real>* v, la::mat<real>* hess, size_t i_start, size_t i_end) {
        for (size_t i=i_start; i<i_end; i++){
            real s1 = 2*(*v)[i]+1;
            real s2 = (*v)[i+1] - (*v)[i] + 1 - (*v)[i]*(*v)[i];
            (*hess)[i][i] = 2*c*(s1*s1 - 2* s2) + 2*c;

            (*hess)[i][i+1] = -2*c*(2*(*v)[i] + 1);
            (*hess)[i+1][i] = -2*c*(2*(*v)[i] + 1);

        }
    }
public:
    static real func(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "fletchcr: n must be positive";

        return function<real>::calculate_value_multithread(&v, fletchcr<real>::calculate_f_job);
    }

    static la::vec<real> gradient(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "fletchcr: n must be positive";

        size_t n = v.size();
        la::vec<real> grad(n);

        function<real>::calculate_gradient_multithread(&v, &grad, fletchcr<real>::calculate_grad_job);
        grad[0] = -2*c*(2*v[0]+1)*(v[1] - v[0] + 1 - v[0]*v[0]);
        grad[n-1] = 2*c*(v[n-1] - v[n-2] + 1 - v[n-2]*v[n-2]);

        return grad;
    }

    static la::mat<real> hessian(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "fletchcr: n must be positive";
        la::mat<real> z(v.size(), v.size(), 0.0);

        size_t n = v.size();
        function<real>::calculate_hessian_multithread(&v, &z, fletchcr<real>::calculate_hessian_job);

        z[0][0] = 2*c*(-1-2*v[0])*(-1-2*v[0]) - 4*c*(1-v[0]-v[0]*v[0]+v[1]);
        z[n-1][n-1] = 2*c;

        return z;
    }

    static la::vec<real> starting_point(const size_t n) {
        if (n == 0)
            throw "fletchcr: n must be positive";
        return la::vec<real>(n, 0.0);
    }

    static function<real> getFunction() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};
template<class real>
const int fletchcr<real>::c;
}
}

#endif //NUMERICALOPTIMIZATION_FLETCHCR_H
