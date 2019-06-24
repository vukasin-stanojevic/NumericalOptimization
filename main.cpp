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
    cerr.precision(16);
    cerr << fixed;


 //   typedef function::ext_rosenbrock<double> func;
//    typedef function::ext_himmelblau<double> func;
//    typedef function::gen_rosenbrock<double> func;
//    typedef function::raydan1<double> func;
//    typedef function::cube<double> func;
//    typedef function::full_hessian2<double> func;
//    typedef function::part_pert_quad<double> func;
//    typedef function::ext_psc1<double> func;
//    typedef function::ext_quad_pen_qp1<double> func;
//    typedef function::almost_pert_quad<double> func;
//    typedef function::diagonal1<double> func;
    typedef function::gen_psc1<double> func;
//    typedef function::fletchcr<double> func;

    const int n = 400;


    map<string, double> params;
//    line_search::binary<double> ls(params);
//    line_search::fixed_line_search<double> ls(params);
    line_search::armijo<double> ls(params);
//    line_search::goldstein<double> ls(params);
//    line_search::wolfe<double> ls(params);
//    line_search::strong_wolfe<double> ls(params);


    method::quasi_newton::dfp<double> opt;
//    method::gradient::momentum<double> opt;
//    method::conjugate_gradient::fletcher_reeves<double> opt;


    auto f = func::getFunction();
    auto x = f.starting_point(n);





    opt::method::method_params<double> methodParams = {f.starting_point(n), 0.1, n, 10000, 1e-7, 1e-16, 0.01, 1};
    //    auto x = la::vec<double>({1,2,3,4,5,6});

//    cerr << "x:" << endl << x << endl;
//    cerr << "func(x):" << endl << func::func(x) << endl;
//    cerr << "grad(x):" << endl << func::gradient(x) << endl;
//    cerr << "hess(x):" << endl << func::hessian(x) << endl; return 0;

    cerr << "Line search parameters:" << endl;
    for (auto e : params) {
        cerr << e.first << " " << e.second << endl;
    }
    cerr << endl;

    opt(f, ls, methodParams);

//    cerr << "xMin: " << x << endl;
    cerr << "fMin: " << opt.get_f_min() << endl;
    cerr << "grNorm: " << opt.get_gr_norm() << endl;
    cerr << "iterNum: " << opt.get_iter_count() << endl;
    cerr << "cpuTime (s): " << opt.get_cpu_time() << endl;
    cerr << "funcEval: " << opt.get_f_call_count() << endl;
    cerr << "gradEval: " << opt.get_g_call_count() << endl;
    cerr << "hessEval: " << opt.get_h_call_count() << endl;

    return 0;
}
