#ifndef EXTENDED_HIMMELBLAU_H_INCLUDED
#define EXTENDED_HIMMELBLAU_H_INCLUDED

#include "function.h"

namespace opt {
namespace function {

template<class real>
class ext_himmelblau {
private:
    static void calculate_f_job(const la::vec<real>* v, std::promise<real>&& prom, size_t i_start, size_t i_end) {
        real z = 0.0;
        real t;

        for (size_t i = i_start; i < i_end; i += 2) {
            t = (*v)[i]*(*v)[i] + (*v)[i+1] - 11;
            z += t*t;
            t = (*v)[i] + (*v)[i+1]*(*v)[i+1] - 7;
            z += t*t;
        }
        prom.set_value(z);
    }

    static void calculate_grad_job(const la::vec<real>* v, la::vec<real>* grad, size_t i_start, size_t i_end) {

        for (size_t i = i_start; i < i_end; i += 2) {
            (*grad)[i] = 4*(*v)[i]*(*v)[i]*(*v)[i] + 2*(*v)[i+1]*(*v)[i+1] + 4*(*v)[i]*(*v)[i+1] - 42*(*v)[i] - 14;
            (*grad)[i+1] = 4*(*v)[i+1]*(*v)[i+1]*(*v)[i+1] + 2*(*v)[i]*(*v)[i] + 4*(*v)[i]*(*v)[i+1] - 26*(*v)[i+1] - 22;
        }
    }

    static void calculate_hessian_job(const la::vec<real>* v, la::mat<real>* hess, size_t i_start, size_t i_end) {

        for (size_t i = i_start; i < i_end; i += 2) {
            (*hess)[i][i] = 12*(*v)[i]*(*v)[i] + 4*(*v)[i+1] - 42;
            (*hess)[i+1][i] = (*hess)[i][i+1] = 4*((*v)[i+1]+(*v)[i]);
            (*hess)[i+1][i+1] = 12*(*v)[i+1]*(*v)[i+1] + 4*(*v)[i] - 22;
        }
    }
public:
    static real func(const la::vec<real>& v) {
        if (v.size() == 0) {
            throw "ext_himmelblau: n must be even and positive";
        }
        return function<real>::calculate_value_multithread(&v, ext_himmelblau<real>::calculate_f_job);
    }

    static la::vec<real> gradient(const la::vec<real>& v) {
        if (v.size() == 0) {
            throw "ext_himmelblau: n must be even and positive";
        }

        size_t n = v.size();
        la::vec<real> z(n, 0.0);

        function<real>::calculate_gradient_multithread(&v, &z, ext_himmelblau<real>::calculate_grad_job);
        return z;
    }

    static la::mat<real> hessian(const la::vec<real>& v) {
        if (v.size() == 0) {
            throw "ext_himmelblau: n must be even and positive";
        }

        size_t n = v.size();
        la::mat<real> z(n, n, 0.0);

        function<real>::calculate_hessian_multithread(&v, &z, ext_himmelblau<real>::calculate_hessian_job);

        return z;
    }

    static la::vec<real> starting_point(const size_t n) {
        if (n == 0) {
            throw "ext_himmelblau: n must be even and positive";
        }

        return la::vec<real>(n, 1.0);
    }

    static function<real> getFunction() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};

}
}

#endif // EXTENDED_HIMMELBLAU_H_INCLUDED
