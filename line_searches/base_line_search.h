#ifndef PROJEKATC___LINE_SEARCH_H
#define PROJEKATC___LINE_SEARCH_H

#include <string>
#include <map>
#include <cmath>
#include "../linear_algebra.h"
#include "../functions/function.h"

namespace opt {
namespace line_search {

template<class real>
class base_line_search{
public:
    virtual real operator()(function::function<real>& f, la::vec<real>& x, la::vec<real>& d) = 0;
protected:
    void rest(std::map<std::string,real>& params, std::map<std::string,real>& custom_params) {
        for (auto e : custom_params){
            params[e.first] = e.second;
        }
    }

    real interp(real t1, real t2, real val1, real val2, real der1, real der2) {
        real d1 = der1+der2-3*(val1-val2)/(t1-t2);
        real d2 = sqrt(d1*d1-der1*der2);
        real tmp = t2 - (t2 - t1)*(der2 + d2 - d1)/(der2 - der1 + 2*d2);
        real t;

        if (tmp >= 0) {
            d2 = sqrt(d1*d1-der1*der2);
            t = t2 - (t2 - t1)*(der2 + d2 - d1)/(der2 - der1 + 2*d2);
        } else {
            t = t1 - 1;
        }

        if (t < t1 || t > t2) {
            t = t1;
        }

        return t;
    };
};

}
}

#endif //PROJEKATC___LINE_SEARCH_H
