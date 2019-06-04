#ifndef PROJEKATC___FLETCHER_REEVES_H
#define PROJEKATC___FLETCHER_REEVES_H

#include "base_method.h"

namespace opt {
namespace method {
namespace conjugate_gradient {

template<class real>
class fletcher_reeves : public base_method<real> {
public:
    fletcher_reeves() : base_method<real>() {}
    void operator()(function::function<real>& f, line_search::base_line_search<real>& ls, la::vec<real>& x) {
        auto p0 = -f.gradient(x);
        real a0 = ls(f, x, p0);
        auto x1 = x + p0*a0;
        auto s0 = p0;

        la::vec<real> gr;
        while (la::norm(gr = f.gradient(x1)) > 1e-7 && this->iter_count++ < 1000) {
            //std::cerr << this->iter_count << ": " << x << "   gnorm = " << la::norm(gr) << '\n';
            auto p1 = -gr; // steepest direction at xn
            auto beta1 = x1.dot(x1) / x.dot(x); // FR beta
            auto s1 = p1 + s0*beta1; // update the conjugate direction
            auto a1 = ls(f, x1, s1);

            auto x2 = x1 + s1*a1;

            x1 = x2;
            s0 = s1;
        }
        x = x1;
    }
};

}
}
}

#endif //PROJEKATC___FLETCHER_REEVES_H
