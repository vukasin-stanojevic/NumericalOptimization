#ifndef PROJEKATC___FIXED_LINE_SEARCH_H
#define PROJEKATC___FIXED_LINE_SEARCH_H

#include "base_line_search.h"

namespace opt {
namespace line_search {

template<class real>
class fixed_step_size : public base_line_search<real> {
private:
    real initial_step; // start point
public:
    fixed_step_size(std::map<std::string, real>& params) {
        std::map<std::string, real> p;
        p["initial_step"] = 0.1;
        this->rest(p, params);
        initial_step = p["initial_step"];
        params = p;
    }

    real operator()(function::function<real>& f, la::vec<real>& x, la::vec<real>& d) {
        la::vec<real> xx = x + d * initial_step;
        this->current_f_val = f(xx);
        this->current_g_val = f.gradient(xx);
        return initial_step;
    }
};

}
}

#endif //PROJEKATC___FIXED_LINE_SEARCH_H
