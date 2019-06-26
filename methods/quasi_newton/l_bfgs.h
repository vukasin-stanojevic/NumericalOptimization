//
// Created by lazar on 25.6.19..
//

#ifndef NUMERICALOPTIMIZATION_L_BFGS_H
#define NUMERICALOPTIMIZATION_L_BFGS_H


#include "../base_method.h"
#include<list>

namespace opt {
    namespace method {
        namespace quasi_newton {

            template<class real>
            class l_bfgs : public base_method<real>{
            private:
                size_t cache_size;
            public:
                l_bfgs() : base_method<real>(),cache_size(5) {}
                l_bfgs(size_t cache_size) : base_method<real>(),cache_size(cache_size) {}
                void operator()(function::function<real>& f, line_search::base_line_search<real>& ls, method_params<real>& params) {
                    this->tic();
                    la::vec<real> x0;
                    la::vec<real> x1 = params.stariting_point;
                    la::vec<real> gradient_prev;
                    la::vec<real> gradient_curr = f.gradient(x1);

                    real H = 1;

                    real fcur = f(x1);
                    real fprev = fcur + 1;

                    std::list<la::vec<real>> s_cache, y_cache;
                    std::list<real> rho_cache;


                    while(la::norm(gradient_curr) > params.epsilon && this->iter_count < params.max_iterations && fabs(fprev - fcur) / (1 + fabs(fcur)) > params.working_precision){


                        la::vec<real> direction = this->two_loop_recursion(H,gradient_curr,s_cache,y_cache,rho_cache);

                        ++this->iter_count;

                        fprev = fcur;
                        x0 = x1;
                        gradient_prev = gradient_curr;

                        real t = ls(f,x1,direction);

                        x1 += direction * t;
                        gradient_curr = f.gradient(x1);
                        fcur = f(x1);

                        //Igi je ovo nesto implementirao
                        // gr1 = f.get_last_g();

                        la::vec<real> s = x1 - x0;
                        this->add_to_cache(s,s_cache);
                        la::vec<real> y = gradient_curr - gradient_prev;
                        this->add_to_cache(y,y_cache);

                        real rho = 1/(s.dot(y));
                        this->add_to_cache(rho,rho_cache);


                        H = (s.dot(y))/(y.dot(y));


                    }




                    this->toc();
                    this->f_min = f(x1);
                    this->gr_norm = la::norm(gradient_curr);
                    this->f_call_count = f.get_call_count();
                    this->g_call_count = f.get_grad_count();
                    this->h_call_count = f.get_hess_count();
                }
            private:
                la::vec<real> two_loop_recursion(real H, la::vec<real>& gradient, std::list<la::vec<real>>& s_cache, std::list<la::vec<real>>& y_cache, std::list<real>& rho_cache){

                    std::list<real> alphas;
                    la::vec<real> q = gradient;


                    auto r_s = s_cache.rbegin();
                    auto r_y = y_cache.rbegin();
                    auto r_rho = rho_cache.rbegin();

                    while(r_s!= s_cache.rend()){
                        real alpha = (*r_rho) * r_s->dot(q);
                        alphas.push_front(alpha);

                        q -= (*r_y)*alpha;

                        ++r_s;
                        ++r_y;
                        ++r_rho;

                    }
                    la::vec<real> direction = q * H;


                    auto s = s_cache.begin();
                    auto y = y_cache.begin();
                    auto rho = rho_cache.begin();
                    auto alpha = alphas.begin();

                    while(s!=s_cache.end()){
                        real beta = (*rho) * y->dot(direction);
                        direction += (*s) * ((*alpha) - beta);

                        ++s;
                        ++y;
                        ++rho;
                        ++alpha;

                    }

                    

                    return direction*(-1);
                }
                template<class T>
                void add_to_cache(T el,std::list<T>& cache){
                    if(cache.size() == this->cache_size){
                        cache.pop_front();
                    }
                    cache.push_back(el);
                }

            };

        }
    }
}


#endif //NUMERICALOPTIMIZATION_L_BFGS_H
