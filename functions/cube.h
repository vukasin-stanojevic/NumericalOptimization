#ifndef CUBE_H_INCLUDED
#define CUBE_H_INCLUDED

#include "function.h"

namespace opt {
namespace function {

template<class real>
class cube {
private:
    static void calculate_f_job(const la::vec<real>* v, std::promise<real>&& prom, size_t i_start, size_t i_end) {
        real z = 0.0;
        real t;
        i_start = i_start == 0 ? 1 : i_start;
        for (size_t i = i_start; i < i_end; ++i) {
            t = (*v)[i] - (*v)[i-1]*(*v)[i-1]*(*v)[i-1];
            z += 100*t*t;
        }
        prom.set_value(z);
    }

    static void calculate_grad_job(const la::vec<real>* v, la::vec<real>* grad, size_t i_start, size_t i_end) {
        i_start = i_start == 0 ? 1 : i_start;
        i_end = i_end == grad->size()? grad->size()-1 : grad->size();
        for (size_t i = i_start; i < i_end; ++i) {
            (*grad)[i] = -600*(*v)[i]*(*v)[i]*(-(*v)[i]*(*v)[i]*(*v)[i] + (*v)[i+1]) + 200*(-(*v)[i-1]*(*v)[i-1]*(*v)[i-1] + (*v)[i]);
        }
    }

    static void calculate_hessian_job(const la::vec<real>* v, la::mat<real>* hess, size_t i_start, size_t i_end) {
        i_end = i_end == v->size()? v->size()-1 : v->size();

        for (size_t i = 0; i < i_end; ++i) {
            (*hess)[i][i] = 200 + 1800*(*v)[i]*(*v)[i]*(*v)[i]*(*v)[i] - 1200*(*v)[i]*(-(*v)[i]*(*v)[i]*(*v)[i] + (*v)[i+1]);
            (*hess)[i][i+1] = (*hess)[i+1][i] = -600*(*v)[i]*(*v)[i];
        }
    }
public:
    static real func(const la::vec<real>& v) {
        if (v.size() == 0) {
            throw "cube: n must be positive";
        }

        real z = (v[0] - 1)*(v[0] - 1);
        z += function<real>::calculate_value_multithread(&v, cube<real>::calculate_f_job);

        return z;
    }

    static la::vec<real> gradient(const la::vec<real>& v) {
        if (v.size() == 0) {
            throw "cube: n must be positive";
        }

        size_t n = v.size();
        size_t m = n - 1;
        la::vec<real> z(n, 0.0);

        z[0] = -2*(1 - v[0]) - 600*v[0]*v[0]*(-v[0]*v[0]*v[0] + v[1]);
        function<real>::calculate_gradient_multithread(&v, &z, cube<real>::calculate_grad_job);
        z[m] = 200*(-v[m-1]*v[m-1]*v[m-1] + v[m]);

        return z;
    }

    static la::mat<real> hessian(const la::vec<real>& v) {
        if (v.size() == 0) {
            throw "cube: n must be positive";
        }

        size_t n = v.size();
        size_t m = n - 1;
        la::mat<real> z(n, n, 0.0);

        function<real>::calculate_hessian_multithread(&v, &z, cube<real>::calculate_hessian_job);

        z[0][0] -= 198;
        z[m][m] = 200;

        return z;
    }

    static la::vec<real> starting_point(const size_t n) {
        if (n == 0) {
            throw "cube: n must be positive";
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

}
}

#endif // CUBE_H_INCLUDED
