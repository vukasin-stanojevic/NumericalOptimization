#ifndef EXTENDED_ROSENBROCK_H_INCLUDED
#define EXTENDED_ROSENBROCK_H_INCLUDED

#include "function.h"

namespace opt {
namespace function {

template<class real>
class ext_rosenbrock {
private:
    static void calculate_f_job(const la::vec<real>* v, std::promise<real>&& prom, size_t i_start, size_t i_end) {
        real z = 0.0;

        for (size_t i = i_start; i < i_end; i += 2) {
            real t = (*v)[i+1] - (*v)[i]*(*v)[i];
            z += c * t*t;
            t = 1 - (*v)[i];
            z += t*t;
        }

        prom.set_value(z);
    }

    static void calculate_grad_job(const la::vec<real>* v, la::vec<real>* grad, size_t i_start, size_t i_end) {
        for (size_t i = i_start; i < i_end; i++) {
            // i & 1 != 0 <=> i is odd
            (*grad)[i] = i & 1 ? c * (2*(*v)[i] - 2*(*v)[i-1]*(*v)[i-1])
                         : c * (4*(*v)[i]*(*v)[i]*(*v)[i] - 4*(*v)[i+1]*(*v)[i]) + 2*(*v)[i] - 2;
        }
    }

    static void calculate_hessian_job(const la::vec<real>* v, la::mat<real>* hess, size_t i_start, size_t i_end) {

        for (size_t i = i_start; i < i_end; i++) {
            // i & 1 != 0 <=> i is odd
            if (i & 1) {
                (*hess)[i][i-1] = -4*c*(*v)[i-1];
                (*hess)[i][i] = 2*c;
            } else {
                (*hess)[i][i] = 12*c*(*v)[i]*(*v)[i] - 4*c*(*v)[i+1] + 2;
                (*hess)[i][i+1] = -4*c*(*v)[i];
            }
        }
    }
public:
    static const int c = 100;

    static real func(const la::vec<real>& v) {
        if (v.size() % 2 || v.size() == 0) {
            throw "ext_rosenbrock: n must be even and positive";
        }

        return function<real>::calculate_value_multithread(&v, ext_rosenbrock<real>::calculate_f_job);
    }

    static la::vec<real> gradient(const la::vec<real>& v) {
        if (v.size() % 2 || v.size() == 0) {
            throw "ext_rosenbrock: n must be even and positive";
        }

        size_t n = v.size();
        la::vec<real> grad(n);

        function<real>::calculate_gradient_multithread(&v, &grad, ext_rosenbrock<real>::calculate_grad_job);

        return grad;
    }

    static la::mat<real> hessian(const la::vec<real>& v) {
        if (v.size() % 2 || v.size() == 0) {
            throw "ext_rosenbrock: n must be even and positive";
        }

        size_t n = v.size();
        la::mat<real> hes(n, n, 0.0);

        function<real>::calculate_hessian_multithread(&v, &hes, ext_rosenbrock<real>::calculate_hessian_job);

        return hes;
    }

    static la::vec<real> starting_point(const size_t n) {
        if (n % 2 || n == 0) {
            throw "ext_rosenbrock: n must be even and positive";
        }

        la::vec<real> z(n, 0.0);

        for (size_t i = 0; i < n; i += 2) {
            z[i] = -1.2;
            z[i+1] = 1;
        }

        return z;
    }

    static function<real> getFunction() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};

template<class real>
const int ext_rosenbrock<real>::c;

}
}

#endif // EXTENDED_ROSENBROCK_H_INCLUDED
