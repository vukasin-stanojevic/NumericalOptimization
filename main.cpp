#include <string>
#include <map>
#include <iostream>
#include "functions.h"
#include "line_searches.h"
#include "methods.h"
#include "matplotlibcpp.h"
#include "test_multithread.h"

using namespace std;
using namespace opt;
namespace plt = matplotlibcpp;

int main() {
	cout.precision(10);
    cout << fixed;
	
    const int n = 10000;

    // typedef opt::function::ext_rosenbrock<double> func;
    // typedef opt::function::gen_psc1<double> func;
    // typedef opt::function::explin1<double> func;
    // typedef opt::function::ext_himmelblau<double> func;
    // typedef opt::function::gen_rosenbrock<double> func;
    // typedef opt::function::raydan1<double> func;
    // typedef opt::function::cube<double> func;
    // typedef opt::function::full_hessian2<double> func;
    // typedef opt::function::part_pert_quad<double> func;
    // typedef opt::function::ext_psc1<double> func;
    // typedef opt::function::ext_quad_pen_qp1<double> func;
    // typedef opt::function::almost_pert_quad<double> func;
    // typedef opt::function::diagonal1<double> func;
    // typedef opt::function::fletchcr<double> func;
    // typedef opt::function::ext_white_and_holst<double> func;
    // typedef opt::function::ext_beale<double> func;
    // typedef opt::function::ext_penalty<double> func;
    // typedef opt::function::hager_function<double> func;
     typedef opt::function::ext_TET<double> func;

    // method::gradient::gradient_descent<double> opt;
    // method::gradient::momentum<double> opt;
    // method::conjugate_gradient::fletcher_reeves<double> opt;
    // method::conjugate_gradient::polak_ribiere<double> opt;
    // method::conjugate_gradient::hestenes_stiefel<double> opt;
    // method::conjugate_gradient::dai_yuan<double> opt;
    // method::conjugate_gradient::cg_descent<double> opt;
    // method::quasi_newton::sr1<double> opt;
    // method::quasi_newton::dfp<double> opt;
    // method::quasi_newton::bfgs<double> opt;
    // method::quasi_newton::l_bfgs<double> opt;
    // method::gradient::nesterov<double> opt;
    // method::gradient::adagrad<double> opt;
    // method::gradient::adadelta<double> opt;
    // method::gradient::rms_prop<double> opt;
    // method::gradient::adam<double> opt;
    // method::gradient::adamax<double> opt;
    // method::gradient::nadam<double> opt;
     method::gradient::amsgrad<double> opt;

    map<string, double> params;
    // line_search::binary<double> ls(params);
     //line_search::fixed_step_size<double> ls(params);
     line_search::armijo<double> ls(params);
    // line_search::goldstein<double> ls(params);
    // line_search::wolfe<double> ls(params);
    // line_search::strong_wolfe<double> ls(params);
    // line_search::approx_wolfe<double> ls(params);

    opt::function::function<double> f = func::getFunction();
    la::vec<double> x = f.starting_point(n);
    // la::vec<double> x({1, 2, 3, 4, 5, 6});

    // cout << "x:" << "\n" << x << "\n";
    // cout << "func(x):" << "\n" << func::func(x) << "\n";
    // cout << "grad(x):" << "\n" << func::gradient(x) << "\n";
    // cout << "hess(x):" << "\n" << func::hessian(x) << "\n"; return 0;

    cout << "n: " << n << "\n";
    cout << "Line search parameters:\n";
    for (auto e : params) {
        cout << e.first << ": " << e.second << "\n";
    }

   opt(f, ls, x);

    cout << "xMin: " << x << "\n";
    cout << "fMin: " << opt.get_f_min() << "\n";
    cout << "grNorm: " << opt.get_gr_norm() << "\n";
    cout << "iterNum: " << opt.get_iter_count() << "\n";
    cout << "cpuTime (s): " << opt.get_cpu_time() << "\n";
    cout << "funcEval: " << opt.get_f_call_count() << "\n";
    cout << "gradEval: " << opt.get_g_call_count() << "\n";
    cout << "hessEval: " << opt.get_h_call_count() << "\n";

    plt::plot(opt.get_gradient_norms());
    std::string s = "l2 norm of gradient through iterations\n";
    s += "method: " + opt.get_method_name();
    plt::title(s);
    plt::show();
    
    // test(&opt, f, ls);

    return 0;
}
