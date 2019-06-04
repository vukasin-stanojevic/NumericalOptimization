#ifndef PROJEKATC___BASE_METHOD_H
#define PROJEKATC___BASE_METHOD_H

#include "../functions/function.h"
#include "../line_searches/base_line_search.h"
#include "../linear_algebra.h"

namespace opt {
namespace method {

template<class real>
class base_method {
public:
    base_method() : iter_count(0) {}

    size_t get_iter_count() const {
        return iter_count;
    }

    virtual void operator()(function::function<real>& f, line_search::base_line_search<real>& ls, la::vec<real>& x) = 0;
protected:
    size_t iter_count;
};

}
}

#endif //PROJEKATC___BASE_METHOD_H
