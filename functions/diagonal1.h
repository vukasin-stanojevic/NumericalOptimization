#ifndef NUMERICALOPTIMIZATION_DIAGONAL1_H
#define NUMERICALOPTIMIZATION_DIAGONAL1_H

#include "function.h"

namespace opt {
namespace function {

template<class real>
class diagonal1 {
private:
    static void calculate_f_job(const la::vec<real>* v, std::promise<real>&& prom, size_t i_start, size_t i_end) {
        real z = 0;
        for (size_t i=i_start; i<i_end; ++i) {
            z += exp((*v)[i]) - (i+1)*(*v)[i];
        }
        prom.set_value(z);
    }

    static void calculate_grad_job(const la::vec<real>* v, la::vec<real>* grad, size_t i_start, size_t i_end) {

        for (size_t i=i_start; i<i_end; ++i) {
            (*grad)[i] = exp((*v)[i]) - (i+1);
        }
    }

    static void calculate_hessian_job(const la::vec<real>* v, la::mat<real>* hess, size_t i_start, size_t i_end) {

        for (size_t i=i_start; i<i_end; ++i) {
            (*hess)[i][i] = exp((*v)[i]);
        }
    }
public:
    static real func(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "diagonal1: n must be positive";
        return function<real>::calculate_value_multithread(&v, diagonal1<real>::calculate_f_job);
    }

    static la::vec<real> gradient(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "diagonal1: n must be positive";
        la::vec<real> z(v.size());

        function<real>::calculate_gradient_multithread(&v, &z, diagonal1<real>::calculate_grad_job);

        return z;
    }

    static la::mat<real> hessian(const la::vec<real>& v) {
        if (v.size() == 0)
            throw "diagonal1: n must be positive";
        la::mat<real> z(v.size(), v.size(), 0.0);

        function<real>::calculate_hessian_multithread(&v, &z, diagonal1<real>::calculate_hessian_job);


        return z;
    }

    static la::vec<real> starting_point(const size_t n) {
        if (n == 0)
            throw "diagonal1: n must be positive";
        return la::vec<real>(n,1.0/n);
    }

    static function<real> getFunction() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};

}
}

#endif //NUMERICALOPTIMIZATION_DIAGONAL1_H
