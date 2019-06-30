#ifndef PROJEKATC___GOLDSTEIN_H
#define PROJEKATC___GOLDSTEIN_H

#include "base_line_search.h"

namespace opt {
namespace line_search {

template<class real>
class goldstein : public base_line_search<real> {
private:
    real steepness; // rho
    real initial_step; // start point
    real gamma;
public:
    goldstein(std::map<std::string, real>& params) {
        std::map<std::string, real> p;
        p["steepness"] = 1e-4;
        p["initial_step"] = 1;
        p["gamma"] = 1.1;
        this->rest(p, params);
        steepness = p["steepness"];
        initial_step = p["initial_step"];
        gamma = p["gamma"];
        params = p;
    }

    real operator()(function::function<real>& f, la::vec<real>& x, la::vec<real>& d) {
        this->iter_count = 0;

        real pad = this->current_g_val.dot(d); // the rate at which f is growing in direction d
        // in other words, it's an approximate value of f(x+d) - f(x)

        real a1 = 0, a2 = 0;
        real a = this->f_values.size() >= 2 ? this->compute_initial_step(this->f_values.end()[-1], this->f_values.end()[-2], this->current_g_val, d) : initial_step;
        bool a2inf = true;
        real f0 = this->current_f_val;
        real ff = f(x + d*a);

        while (this->iter_count < 52) {
            ++this->iter_count;

            if (ff > f0 + steepness*a*pad) {
                a2 = a;
                a2inf = false;
                a = (a1 + a2) / 2;
            } else if (ff < f0 + (1-steepness)*a*pad) {
                a1 = a;
                if (!a2inf) {
                    a = (a1 + a2) / 2;
                } else {
                    a *= gamma;
                }
            } else {
                break;
            }

            ff = f(x + d*a);
        }

        this->current_f_val = ff;
        this->current_g_val = f.gradient(x + d * a);
        return a;
    }
};

}
}

#endif //PROJEKATC___GOLDSTEIN_H
