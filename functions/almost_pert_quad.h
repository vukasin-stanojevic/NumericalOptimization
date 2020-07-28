#ifndef NUMERICALOPTIMIZATION_AP_QUAD_H
#define NUMERICALOPTIMIZATION_AP_QUAD_H

#include "function.h"

namespace opt {
namespace function {

template<class real>
class almost_pert_quad {
private:
    static void calculate_f_job(const la::vec<real>* v, std::promise<real>&& prom, size_t i_start, size_t i_end) {
        real z = 0;
        real t;
        for (size_t i=i_start; i<i_end; ++i) {
            t = (i+1)*(*v)[i]*(*v)[i];
            z += t;
        }
        prom.set_value(z);
    }

    static void calculate_grad_job(const la::vec<real>* v, la::vec<real>* grad, size_t i_start, size_t i_end) {

        for (size_t i=i_start; i<i_end; ++i) {
            (*grad)[i] = 2*(i+1)*(*v)[i];
        }
    }

    static void calculate_hessian_job(const la::vec<real>* v, la::mat<real>* hess, size_t i_start, size_t i_end) {

        for (size_t i=i_start; i<i_end; ++i) {
            (*hess)[i][i] = 2*(i+1);
        }
    }
public:
    static const int c = 100;

    static real func(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "almost_pert_quad: n must be positive";
        real z = function<real>::calculate_value_multithread(&v, almost_pert_quad<real>::calculate_f_job);
        size_t n = v.size();
        z+=n*(v[0]+v[n-1])*(v[0]+v[n-1])/c;

        return z;
    }

    static la::vec<real> gradient(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "almost_pert_quad: n must be positive";
        size_t n = v.size();
        la::vec<real> grad(n);

        function<real>::calculate_gradient_multithread(&v, &grad, almost_pert_quad<real>::calculate_grad_job);

        real t = (2.0/c)*(v[0]+v[n-1]);

        grad[0]+= n*t;
        grad[n-1]+= n*t;

        return grad;
    }

    static la::mat<real> hessian(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "almost_pert_quad: n must be positive";
        la::mat<real> z(v.size(), v.size(), 0.0);
        size_t n = v.size();

        function<real>::calculate_hessian_multithread(&v, &z, almost_pert_quad<real>::calculate_hessian_job);

        z[n-1][n-1] += (2.0*n)/c;
        z[0][0] += (2.0*n)/c;
        z[n-1][0] += (2.0*n)/c;
        z[0][n-1] += (2.0*n)/c;

        return z;
    }

    static la::vec<real> starting_point(const size_t n) {
        if (n == 0)
            throw "almost_pert_quad: n must be positive";
        return la::vec<real>(n,0.5);
    }

    static function<real> getFunction() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};
template<class real>
const int almost_pert_quad<real>::c;
}
}

#endif //NUMERICALOPTIMIZATION_AP_QUAD_H
