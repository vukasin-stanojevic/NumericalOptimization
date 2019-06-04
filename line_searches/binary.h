#ifndef PROJEKATC___BINARY_H
#define PROJEKATC___BINARY_H

#include "base_line_search.h"

namespace opt {
namespace line_search {

template<class real>
class binary : public base_line_search<real> {
private:
    real initial_step;
public:
    binary(std::map<std::string, real>& params) {
        std::map<std::string, real> p;
        p["initial_step"] = 1;
        this->rest(p, params);
        initial_step = p["initial_step"];
        params = p;
    }

    real operator()(function::function<real>& func, la::vec<real>& x, la::vec<real>& d) {
        real a = initial_step;
        // real fstart = f(x);
        real f0 = func(x + d * a);
        real f1 = func(x + d * a * 2);

        if (f1 < f0) {
            a *= 2;
            real curr = f1;
            real t = func(x + d*a*2);
            while (t < curr) {
                curr = t;
                a *= 2;
                t = func(x + d*a*2);
            }
            return a;
        } else {
            real curr = f0;
            real t = func(x + d*a/2);
            while (t < curr) {
                a /= 2;
                curr = t;
                t = func(x + d*a/2);
            }
            return a;
        }
    }
};

}
}

#endif //PROJEKATC___BINARY_H
