#ifndef PROJEKATC___GRADIENT_DESCENT_H
#define PROJEKATC___GRADIENT_DESCENT_H

#include "base_method.h"

namespace opt {
namespace method {
namespace gradient {

template<class real>
class gradient_descent : public base_method<real> {
public:
    gradient_descent() : base_method<real>() {}
    void operator()(function::function<real>& f, line_search::base_line_search<real>& ls, la::vec<real>& x) {
        la::vec<real> gr = f.gradient(x);
        while (la::norm(gr) > 1e-8 && this->iter_count < 10000) {
            la::vec<real> d = -gr;
            x += d * ls(f, x, d);

            gr = f.gradient(x);
            ++this->iter_count;
        }
    }
};

}
}
}

#endif //PROJEKATC___GRADIENT_DESCENT_H
