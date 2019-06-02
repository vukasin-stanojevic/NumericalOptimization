#include <string>
#include<map>
#include<iostream>
#include "functions.h"
#include "line_searches.h"
#include "methods.h"

using namespace std;


int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);
    cerr.tie(nullptr);


    auto function = functions::ap_quad<double>::getFunction();



    auto x0 = function.starting_point(4);
    cerr.precision(8);
    cerr << fixed;

    cout << x0 << endl;
    cout << function.hessian(x0) << endl;






    map<string, double> params;
    line_searches::line_search<double>* line_search = new line_searches::strong_wolfe<double>(params);




    opt_methods::conjugate_gradient::fletcher_reeves<double> opt;

    opt(function,*line_search,x0);


    cerr << "Params:" << endl;
    for(auto e : params){
        cerr << e.first << " " << e.second << endl;
    }


    return 0;
}