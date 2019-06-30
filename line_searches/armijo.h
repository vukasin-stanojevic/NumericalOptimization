#ifndef PROJEKATC___ARMIJO_H
#define PROJEKATC___ARMIJO_H

#include "base_line_search.h"

namespace opt {
namespace line_search {

template<class real>
class armijo : public base_line_search<real> {
private:
    real steepness; // rho
    real initial_step; // start point
public:
    armijo(std::map<std::string, real>& params) {
        std::map<std::string, real> p;
        p["steepness"] = 1e-4;
        p["initial_step"] = 1;
        this->rest(p, params);
        steepness = p["steepness"];
        initial_step = p["initial_step"];
        params = p;
    }

    real operator()(function::function<real>& f, la::vec<real>& x, la::vec<real>& d) {
        this->iter_count = 0;

        real f0 = this->current_f_val; // value in starting point
        real pad = this->current_g_val.dot(d); // the rate at which f is growing in direction d
        // in other words, it's an approximate value of f(x+d) - f(x)

        real a_curr = this->f_values.size() >= 2 ? this->compute_initial_step(this->f_values.end()[-1], this->f_values.end()[-2], this->current_g_val, d) : initial_step;

        real f_curr, f_prev, a_prev;
        f_curr = f(x + d * a_curr);

        while (f_curr > f0 - steepness * a_curr * pad) {
            ++this->iter_count;

            real a_new;
            if (this->iter_count == 1) {
                // find next point using quadratic interpolation
                a_new = pad * a_curr * a_curr / 2 / (f0 - f_curr + pad * a_curr);
            } else {
                // find next point using cubic interpolation
                real cubic = a_prev * a_prev * (f_curr - f0);
                cubic -= a_curr * a_prev * a_prev * pad;
                cubic += a_curr * a_curr * (f0 - f_prev + a_prev * pad);
                cubic /= a_curr * a_curr * (a_curr - a_prev) * a_prev * a_prev;

                real quadr = -cubic * a_curr * a_curr * a_curr - f0 + f_curr - a_curr * pad;
                quadr /= a_curr * a_curr;

                a_new = (-quadr + sqrt(quadr * quadr - 3 * cubic * pad)) / (3 * cubic);
            }

            a_prev = a_curr;
            a_curr = a_new;

            f_prev = f_curr;
            f_curr = f(x + d * a_curr);
        }

        this->current_f_val = f_curr;
        this->current_g_val = f.gradient(x + d * a_curr);
        return a_curr;
    }
};

}
}

#endif //PROJEKATC___ARMIJO_H
