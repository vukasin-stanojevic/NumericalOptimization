#ifndef GENERALIZED_ROSENBROCK_H_INCLUDED
#define GENERALIZED_ROSENBROCK_H_INCLUDED

#include "function.h"

namespace opt {
namespace function {

template<class real>
class gen_rosenbrock {
private:
    static void calculate_f_job(const la::vec<real>* v, std::promise<real>&& prom, size_t i_start, size_t i_end) {
        i_end = i_end == v->size() ? i_end - 1 : i_end;
        real z = 0.0;
        real t;

        for (size_t i = i_start; i < i_end; ++i) {
            t = (*v)[i+1] - (*v)[i]*(*v)[i];
            z += c * t*t;
            t = 1 - (*v)[i];
            z += t*t;
        }

        prom.set_value(z);
    }

    static void calculate_grad_job(const la::vec<real>* v, la::vec<real>* grad, size_t i_start, size_t i_end) {
        i_end = i_end == v->size() ? i_end - 1 : i_end;
        i_start = i_start == 0 ? 1 : i_start;

        for (size_t i = i_start; i < i_end; ++i) {
            (*grad)[i] = c * (4*(*v)[i]*(*v)[i]*(*v)[i] - 2*(*v)[i-1]*(*v)[i-1] - 4*(*v)[i+1]*(*v)[i] + 2*(*v)[i]) - 2*(1-(*v)[i]);
        }
    }

    static void calculate_hessian_job(const la::vec<real>* v, la::mat<real>* hess, size_t i_start, size_t i_end) {
        i_end = i_end == v->size() ? i_end - 1 : i_end;
        i_start = i_start == 0 ? 1 : i_start;

        for (size_t i = i_start; i < i_end; i++) {
            (*hess)[i][i-1] = -4*c*(*v)[i-1];
            (*hess)[i][i] = 2 + 2*c + 8*c*(*v)[i]*v[i] - 4*c*(-(*v)[i]*(*v)[i] + (*v)[i+1]);
            (*hess)[i][i+1] = -4*c*(*v)[i];
        }
    }
public:
    static const int c = 100;

    static real func(const la::vec<real>& v) {
        if (v.size() == 0) {
            throw "gen_rosenbrock: n must be positive";
        }

        return function<real>::calculate_value_multithread(&v, gen_rosenbrock<real>::calculate_f_job);
    }

    static la::vec<real> gradient(const la::vec<real>& v) {
        if (v.size() == 0) {
            throw "gen_rosenbrock: n must be positive";
        }

        size_t n = v.size();
        size_t m = n - 1;
        la::vec<real> z(n, 0.0);

        z[0] = c * 4 * (v[0]*v[0]*v[0] - v[0]*v[1]) - 2*(1-v[0]);
        function<real>::calculate_gradient_multithread(&v, &z, gen_rosenbrock<real>::calculate_grad_job);
        z[m] = c * 2 * (v[m] - v[m-1]*v[m-1]);

        return z;
    }

    static la::mat<real> hessian(const la::vec<real>& v) {
        if (v.size() == 0) {
            throw "gen_rosenbrock: n must be positive";
        }

        size_t n = v.size();
        size_t m = n - 1;
        la::mat<real> z(n, n, 0.0);

        // computes first row
        z[0][0] = 2 + 8*c*v[0]*v[0] - 4*c*(-v[0]*v[0] + v[1]);
        z[0][1] = -4*c*v[0];

        // computes all rows except for first and last
        function<real>::calculate_hessian_multithread(&v, &z, gen_rosenbrock<real>::calculate_hessian_job);

        // computes last row
        z[m][m] = 2*c;
        z[m][m-1] = -4*c*v[m-1];

        return z;
    }

    static la::vec<real> starting_point(const size_t n) {
        if (n == 0) {
            throw "gen_rosenbrock: n must be positive";
        }

        la::vec<real> z(n, 0.0);

        for (size_t i = 0; i < n; ++i) {
            z[i] = i & 1 ? 1 : -1.2; // i & 1 != 0 <=> i is odd
        }

        return z;
    }

    static function<real> getFunction() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};

template<class real>
const int gen_rosenbrock<real>::c;

}
}

#endif // GENERALIZED_ROSENBROCK_H_INCLUDED
