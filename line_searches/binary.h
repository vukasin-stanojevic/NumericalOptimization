#ifndef PROJEKATC___BINARY_H
#define PROJEKATC___BINARY_H

#include "base_line_search.h"

namespace opt {
namespace line_search {

template<class real>
class binary : public base_line_search<real> {
private:
    real initial_step; // start point
public:
    binary(std::map<std::string, real>& params) {
        std::map<std::string, real> p;
        p["initial_step"] = 1;
        this->rest(p, params);
        initial_step = p["initial_step"];
        params = p;
    }

    real operator()(function::function<real>& f, la::vec<real>& x, la::vec<real>& d) {
        this->iter_count = 0;

        real a = this->f_values.size() >= 2 ? this->compute_initial_step(this->f_values.end()[-1], this->f_values.end()[-2], this->current_g_val, d) : initial_step;
        // real fstart = f(x);
        real f0 = f(x + d * a);
        real f1 = f(x + d * a * 2);

        if (f1 < f0) {
            a *= 2;
            real curr = f1;
            real t = f(x + d*a);
            while (t < curr) {
                ++this->iter_count;
                a *= 2;
                curr = t;
                t = f(x + d*a);
            }
            this->current_f_val = t;
            this->current_g_val = f.gradient(x + d*a);
            return a;
        } else {
            a /= 2;
            real curr = f0;
            real t = f(x + d*a);
            while (t < curr) {
                ++this->iter_count;
                a /= 2;
                curr = t;
                t = f(x + d*a);
            }
            this->current_f_val = t;
            this->current_g_val = f.gradient(x + d*a);
            return a;
        }
    }
};

}
}

#endif //PROJEKATC___BINARY_H
