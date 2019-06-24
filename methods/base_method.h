#ifndef PROJEKATC___BASE_METHOD_H
#define PROJEKATC___BASE_METHOD_H

#include <chrono>
#include "../functions/function.h"
#include "../line_searches/base_line_search.h"
#include "../linear_algebra.h"

namespace opt {
namespace method {
template<class real>
struct method_params{
    la::vec<real> stariting_point;
    real step_size;
    size_t dimensionality;
    size_t max_iterations;
    real epsilon;
    real working_precision;
    real min_step_size;
    real StartingPoint;
    real nu = 0.1; // threshold for restarting beta in CG methods
};
template<class real>
class base_method {
public:
    base_method() : iter_count(0), f_call_count(0), g_call_count(0),
                    h_call_count(0), gr_norm(-1), f_min(0), cpu_time(0) {}

    virtual void operator()(function::function<real>& f, line_search::base_line_search<real>& ls, method_params<real>& params) = 0;

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
    size_t iter_count;
    size_t f_call_count;
    size_t g_call_count;
    size_t h_call_count;
    double cpu_time;
    real gr_norm;
    real f_min;

    void tic() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void toc() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dsec = end_time - start_time;
        cpu_time = dsec.count();
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};


}
}

#endif //PROJEKATC___BASE_METHOD_H
