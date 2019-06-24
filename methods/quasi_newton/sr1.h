//
// Created by Zucchabar Inc on 6/23/2019.
//

#ifndef NUMERICALOPTIMIZATION_SR1_H
#define NUMERICALOPTIMIZATION_SR1_H

#include "../base_method.h"

namespace opt {
namespace method {
namespace quasi_newton {

template<class real>
class sr1 : public base_method<real>{
public:
    sr1() : base_method<real>() {}
    void operator()(function::function<real>& f, line_search::base_line_search<real>& ls, method_params<real>& params) {
        this->tic();
        la::vec<real> x0;
        la::vec<real> x1 = params.stariting_point;
        la::vec<real> gr0;
        la::vec<real> gr1 = f.gradient(x1);

        la::mat<real> H = la::mat<real>::id(params.dimensionality);

        real r = 1e-8; //coef which determine to make update of H or not

        real fcur = f(x1);
        real fprev = fcur + 1;

        while(la::norm(gr1) > params.epsilon && this->iter_count < params.max_iterations && fabs(fprev - fcur) / (1 + fabs(fcur)) > params.working_precision){
            la::vec<real> direction = (H.dot(gr1))*(-1);
            ++this->iter_count;

            fprev = fcur;
            x0 = x1;
            gr0 = gr1;

            real t = ls(f,x1,direction);

            x1 += direction * t;
            gr1 = f.gradient(x1);
            fcur = f(x1);

            //Igi je ovo nesto implementirao
            // gr1 = f.get_last_g();

            la::vec<real> s = x1 - x0;
            la::vec<real> y = gr1 - gr0;

            la::vec<real> tmp_s_H_dot_y = s - H.dot(y);

            if(tmp_s_H_dot_y.dot(y) >= r * la::norm(y) * la::norm(tmp_s_H_dot_y) ){
                H +=  (tmp_s_H_dot_y.outer(tmp_s_H_dot_y))/(tmp_s_H_dot_y.outer(y));
                cout << "uso";
            }




        }




        this->toc();
        this->f_min = f(x1);
        this->gr_norm = la::norm(gr1);
        this->f_call_count = f.get_call_count();
        this->g_call_count = f.get_grad_count();
        this->h_call_count = f.get_hess_count();
    }

};

}
}
}

#endif //NUMERICALOPTIMIZATION_SR1_H
