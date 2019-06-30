#ifndef PROJEKATC___BASE_METHOD_H
#define PROJEKATC___BASE_METHOD_H

#include <chrono>
#include "../functions/function.h"
#include "../line_searches/base_line_search.h"
#include "../utilities/linear_algebra.h"

namespace opt {
namespace method {

template<class real>
class base_method {
public:
    base_method(real epsilon = 1e-6, size_t max_iter = 10000, real working_precision = 1e-16)
                  : iter_count(0), f_call_count(0), g_call_count(0),
                    h_call_count(0), gr_norm(-1), f_min(0), cpu_time(0),
                    epsilon(epsilon), max_iter(max_iter), working_precision(working_precision) {}

    virtual void operator()(function::function<real>& f, line_search::base_line_search<real>& ls, la::vec<real>& x) = 0;

    real get_epsilon() const {
        return epsilon;
    }

    size_t get_max_iter() const {
        return max_iter;
    }

    real get_working_precision() const {
        return working_precision;
    }

    real get_f_min() const {
        return f_min;
    }

    real get_gr_norm() const {
        return gr_norm;
    }

    double get_cpu_time() const {
        return cpu_time;
    }

    size_t get_iter_count() const {
        return iter_count;
    }

    size_t get_f_call_count() const {
        return f_call_count;
    }

    size_t get_g_call_count() const {
        return g_call_count;
    }

    size_t get_h_call_count() const {
        return h_call_count;
    }
protected:
    real epsilon; // stops the method if gradient norm is <= epsilon
    size_t max_iter; // maximum number of iterations in the outer loop
    real working_precision; // stops the method if optimization becomes too slow

    size_t iter_count; // number of iterations in the method (outer) loop
    size_t f_call_count; // number of function evaluations
    size_t g_call_count; // number of gradient evaluations
    size_t h_call_count; // number of hessian evaluations
    double cpu_time; // total method cpu time in seconds
    real gr_norm; // resulting gradient norm
    real f_min; // resulting minimal function value

    // Captures the start time of the method. Should be called before toc() and
    // at the beginning of the method body.
    void tic() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    // Calculates cpu time based on start_time. Should be called after tic() and
    // at the end of the method body.
    void toc() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dsec = end_time - start_time;
        cpu_time = dsec.count();
    }
private:
    // used to calculate cpu time
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};

}
}

#endif //PROJEKATC___BASE_METHOD_H
