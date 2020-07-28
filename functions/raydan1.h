#ifndef RAYDAN1_H_INCLUDED
#define RAYDAN1_H_INCLUDED

#include <cmath>
#include "function.h"

namespace opt {
namespace function {

template<class real>
class raydan1 {
private:
    static void calculate_f_job(const la::vec<real>* v, std::promise<real>&& prom, size_t i_start, size_t i_end) {
        real z = 0.0;

        for (size_t i = i_start; i < i_end; ++i) {
            z += ((i+1) / 10.0) * (exp((*v)[i]) - (*v)[i]);
        }

        prom.set_value(z);
    }

    static void calculate_grad_job(const la::vec<real>* v, la::vec<real>* grad, size_t i_start, size_t i_end) {
        for (size_t i = i_start; i < i_end; ++i) {
            (*grad)[i] = ((i+1) / 10.0) * (exp((*v)[i]) - 1);
        }
    }

    static void calculate_hessian_job(const la::vec<real>* v, la::mat<real>* hess, size_t i_start, size_t i_end) {
        for (size_t i = i_start; i < i_end; ++i) {
            (*hess)[i][i] = ((i+1) / 10.0) * exp((*v)[i]);
        }
    }
public:
    static real func(const la::vec<real>& v) {
        if (v.size() == 0) {
            throw "raydan1: n must be positive";
        }

        return function<real>::calculate_value_multithread(&v, raydan1<real>::calculate_f_job);
    }

    static la::vec<real> gradient(const la::vec<real>& v) {
        if (v.size() == 0) {
            throw "raydan1: n must be positive";
        }

        la::vec<real> z(v.size());
        function<real>::calculate_gradient_multithread(&v, &z, raydan1<real>::calculate_grad_job);
        return z;
    }

    static la::mat<real> hessian(const la::vec<real>& v) {
        if (v.size() == 0) {
            throw "raydan1: n must be positive";
        }

        size_t n = v.size();
        la::mat<real> z(n, n, 0.0);
        function<real>::calculate_hessian_multithread(&v, &z, raydan1<real>::calculate_hessian_job);

        return z;
    }

    static la::vec<real> starting_point(const size_t n) {
        if (n == 0) {
            throw "raydan1: n must be positive";
        }

        return la::vec<real>(n, 1.0);
    }

    static function<real> getFunction() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};

}
}

#endif // RAYDAN1_H_INCLUDED
