#include <string>
#include <map>
#include <iostream>
#include "functions.h"
#include "line_searches.h"
#include "methods.h"

using namespace std;
using namespace opt;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);
    cerr.tie(nullptr);
    cerr.precision(8);
    cerr << fixed;


//    typedef function::ap_quad<double> func;
    typedef function::pp_quad<double> func;
//    typedef function::extended_rosenbrock<double> func;


    map<string, double> params;
//    line_search::binary<double> ls(params);
//    line_search::fixed_line_search<double> ls(params);
//    line_search::armijo<double> ls(params);
//    line_search::goldstein<double> ls(params);
//    line_search::wolfe<double> ls(params);
    line_search::strong_wolfe<double> ls(params);


//    method::gradient::gradient_descent<double> opt;
//    method::gradient::momentum<double> opt;
    method::conjugate_gradient::fletcher_reeves<double> opt;


    auto f = func::getFunction();
    auto x = f.starting_point(4);
    cerr << "x:" << endl << x << endl;
    cerr << "func(x):" << endl << func::func(x) << endl;
    cerr << "grad(x):" << endl << func::gradient(x) << endl;
    cerr << "hess(x):" << endl << func::hessian(x) << endl;
    cerr << "Line search params:" << endl;
    for (auto e : params) {
        cerr << e.first << " " << e.second << endl;
    }
    cerr << endl;

    opt(f, ls, x);

    cerr << "Xmin: " << x << endl;
    cerr << "Fmin: " << func::func(x) << endl;
    cerr << "grNorm: " << la::norm(func::gradient(x)) << endl;
    cerr << "iterNum: " << opt.get_iter_count() << endl;
    cerr << "funcEval: " << f.get_call_count() << endl;
    cerr << "gradEval: " << f.get_grad_count() << endl;

    return 0;
}
