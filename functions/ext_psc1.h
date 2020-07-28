#ifndef PROJEKATC___EXTENDED_PSC1_H
#define PROJEKATC___EXTENDED_PSC1_H

#include "function.h"

namespace opt {
namespace function {

template<class real>
class ext_psc1 {
private:
    static void calculate_f_job(const la::vec<real>* v, std::promise<real>&& prom, size_t i_start, size_t i_end) {
        real z = 0.0;
        real t;
        for (size_t i=i_start; i<i_end; i+=2) {
            t = (*v)[i]*(*v)[i] + (*v)[i+1]*(*v)[i+1] + (*v)[i]*(*v)[i+1];
            z += t*t;
            t = sin((*v)[i]);
            z += t*t;
            t = cos((*v)[i+1]);
            z += t*t;
        }

        prom.set_value(z);
    }

    static void calculate_grad_job(const la::vec<real>* v, la::vec<real>* grad, size_t i_start, size_t i_end) {
        real t;

        for (size_t i=i_start; i<i_end; i+=2) {
            t = (*v)[i]*(*v)[i] + (*v)[i+1]*(*v)[i+1] + (*v)[i]*(*v)[i+1];
            (*grad)[i] += 2*t*(2*(*v)[i] + (*v)[i+1]);
            (*grad)[i] += 2*sin((*v)[i])*cos((*v)[i]);

            (*grad)[i+1] += 2*t*(2*(*v)[i+1] + (*v)[i]);
            (*grad)[i+1] -= 2*cos((*v)[i+1])*sin((*v)[i+1]);
        }
    }

    static void calculate_hessian_job(const la::vec<real>* v, la::mat<real>* hess, size_t i_start, size_t i_end) {

        for (size_t i=i_start; i<i_end; i+=2) {
            // 0-0
            real a = (*v)[i];
            real b = (*v)[i+1];

            (*hess)[i][i] = 2*(2*a+b)*(2*a+b) + 4*(a*a+a*b+b*b)
                      + 2*cos(a)*cos(a) - 2*sin(a)*sin(a);

            // 0-1
            (*hess)[i+1][i] = 2*(2*a+b)*(a+2*b) + 2*(a*a+a*b+b*b);
            (*hess)[i][i+1] = (*hess)[i+1][i];

            // 1-1
            (*hess)[i+1][i+1] = 2*(a+2*b)*(a+2*b) + 4*(a*a+a*b+b*b)
                          - 2*cos(b)*cos(b) + 2*sin(b)*sin(b);
        }
    }
public:
    static real func(const la::vec<real>& v) {
        if (v.size() % 2 || v.size() == 0)
            throw "ext_psc1: n must be even and positive";

        return function<real>::calculate_value_multithread(&v, ext_psc1<real>::calculate_f_job);
    }

    static la::vec<real> gradient(const la::vec<real>& v) {
        if (v.size() % 2 || v.size() == 0)
            throw "ext_psc1: n must be even and positive";
        la::vec<real> z(v.size(), 0.0);
        function<real>::calculate_gradient_multithread(&v, &z, ext_psc1<real>::calculate_grad_job);

        return z;
    }

    static la::mat<real> hessian(const la::vec<real>& v) {
        if (v.size() % 2 || v.size() == 0)
            throw "ext_psc1: n must be even and positive";
        la::mat<real> z(v.size(), v.size(), 0.0);

        function<real>::calculate_hessian_multithread(&v, &z, ext_psc1<real>::calculate_hessian_job);

        return z;
    }

    static la::vec<real> starting_point(const size_t n) {
        if (n % 2 || n == 0)
            throw "ext_psc1: n must be even and positive";
        la::vec<real> z(n, 0);
        for (size_t i=0; i<n; i+=2) {
            z[i] = 3;
            z[i+1] = 0.1;
        }
        return z;
    }

    static function<real> getFunction() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};

}
}

#endif //PROJEKATC___EXTENDED_PSC1_H
