#ifndef PROJEKATC___EXPLIN1_H
#define PROJEKATC___EXPLIN1_H

#include "function.h"

namespace opt {
namespace function {

template<class real>
class explin1 {
private:
    static void calculate_f_job(const la::vec<real>* v, std::promise<real>&& prom, size_t i_start, size_t i_end) {
        real z = 0.0;
        size_t i_end_2 = i_end == v->size()? i_end - 1 : i_end;

        for (size_t i=i_start; i<i_end_2; i++)
            z += exp(0.1 * (*v)[i] * (*v)[i+1]);
        for (size_t i=i_start; i<i_end; i++)
            z -= (*v)[i] * 10 * (i+1);

        prom.set_value(z);
    }

    static void calculate_grad_job(const la::vec<real>* v, la::vec<real>* grad, size_t i_start, size_t i_end) {
        size_t i_end_2 = i_end == v->size()? i_end - 1 : i_end;

        for (size_t i=i_start; i<i_end_2; i++) {
            (*grad)[i] += exp((*v)[i]*(*v)[i+1] / 10) * (*v)[i+1] / 10;
            (*grad)[i+1] += exp((*v)[i]*(*v)[i+1] / 10) * (*v)[i] / 10;
        }
        for (size_t i=i_start; i<i_end; i++)
            (*grad)[i] -= (real)10*(i+1);
    }

    static void calculate_hessian_job(const la::vec<real>* v, la::mat<real>* hess, size_t i_start, size_t i_end) {
        size_t i_end_2 = i_end == v->size()? i_end - 1 : i_end;
        real a;
        real b;
        real c;

        for (size_t i=i_start; i<i_end; i++) {
            b = (*v)[i];
            if (i > 0) {
                a = (*v)[i-1];
                (*hess)[i][i] += a*a*exp(0.1*a*b);
            }
            if (i+1 < v->size()) {
                c = (*v)[i+1];
                (*hess)[i][i] += c*c*exp(0.1*b*c);
            }
        }

        for (size_t i=i_start; i<i_end_2; i++) {
            a = (*v)[i];
            b = (*v)[i+1];
            (*hess)[i][i+1] = (10+a*b)*exp(0.1*a*b);
            (*hess)[i+1][i] = (*hess)[i][i+1];
        }
    }
public:
    static real func(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "explin1: n must be positive";
        return function<real>::calculate_value_multithread(&v, explin1<real>::calculate_f_job);
    }

    static la::vec<real> gradient(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "explin1: n must be positive";
        la::vec<real> z(v.size(), 0.0);
        function<real>::calculate_gradient_multithread(&v, &z, explin1<real>::calculate_grad_job);

        return z;
    }

    static la::mat<real> hessian(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "explin1: n must be positive";
        la::mat<real> z(v.size(), v.size(), 0.0);

        function<real>::calculate_hessian_multithread(&v, &z, explin1<real>::calculate_hessian_job);

        return z / (real)100;
    }

    static la::vec<real> starting_point(const size_t n) {
        if (n == 0)
            throw "explin1: n must be positive";
        return la::vec<real>(n, 0.0);
    }

    static function<real> getFunction() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};

}
}

#endif //PROJEKATC___EXPLIN1_H
