//
// Created by lazar on 26.5.19..
//

#ifndef PROJEKATC___BASE_METHOD_H
#define PROJEKATC___BASE_METHOD_H

#include<map>
#include<string>

#include "../linear_algebra.h"
#include "../functions.h"
#include "../line_searches.h"

using namespace std;
namespace opt_methods {
    template<class method,class real>
    class base_method {
    protected:
        size_t steps;
    public:
        base_method():steps(0){}
        size_t get_number_steps(){
            return steps;
        }

        template<class line_search,class function>
        void operator()(function &func, line_search &lin_sr, la::vec<real> &x0){
            static_cast<method*>(this)->operator()(func,lin_sr,x0);
        }

    };
}

#endif //PROJEKATC___BASE_METHOD_H
