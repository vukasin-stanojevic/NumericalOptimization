//
// Created by lazar on 26.5.19..
//

#ifndef PROJEKATC___FIXED_LINE_SEARCH_H
#define PROJEKATC___FIXED_LINE_SEARCH_H

#include "line_search.h"


namespace line_searches {
    template<class real>
    class fixed_line_search : public line_search<real> {
    private:
        real steepness;
        real initial_step;
    public:
        fixed_line_search(map<string, real>& params){
            map<string,real> p;
            p["initial_step"] = 1;
            rest(p,params);
            initial_step = p["initial_step"];
            params = p;

        }
        real operator()(vec<real> &x0, vec<real> &d, functions::function<real> &func) {
            return initial_step;
        }
    };
}

#endif //PROJEKATC___FIXED_LINE_SEARCH_H
