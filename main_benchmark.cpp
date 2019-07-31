#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include "functions.h"
#include "line_searches.h"
#include "methods.h"

using namespace std;
using namespace opt;

// select a function to optimize
typedef function::full_hessian2<double> func;

// a helper function to calculate the output filename
string calc_filename(method::base_method<double>* opt, line_search::base_line_search<double>* ls, int n) {
    string methodtype = typeid(*opt).name();
    string lstype = typeid(*ls).name();

    string s = "cpp_";
    
    if (typeid(func) == typeid(function::gen_rosenbrock<double>)) {
        s += "genrosenbrock";
    } else if (typeid(func) == typeid(function::ext_rosenbrock<double>)) {
        s += "extrosenbrock";
    } else if (typeid(func) == typeid(function::diagonal1<double>)) {
        s += "diagonal1";
    } else if (typeid(func) == typeid(function::full_hessian2<double>)) {
        s += "fullhessian2";
    }
    
    s += "_" + to_string(n) + "_";

    if (methodtype.find("cg_descent") != string::npos) {
        s += "cgd";
    } else if (methodtype.find("dai_yuan") != string::npos) {
        s += "dy";
    } else if (methodtype.find("fletcher_reeves") != string::npos) {
        s += "fr";
    } else if (methodtype.find("hestenes_stiefel") != string::npos) {
        s += "hs";
    } else if (methodtype.find("polak_ribiere") != string::npos) {
        s += "pr";
    }

    s += "_";

    if (lstype.find("approx_wolfe") != string::npos) {
        s += "approxwolfe";
    } else if (lstype.find("strong_wolfe") != string::npos) {
        s += "strongwolfe";
    }

    s += ".txt";

    return s;
}

int main() {
    cout.precision(10);
    cout << fixed;

    const int t = 10; // number of repeated tests

    // select a list of methods to benchmark test
    vector<method::base_method<double>*> methods;
    // methods.push_back(new method::conjugate_gradient::fletcher_reeves<double>());
    // methods.push_back(new method::conjugate_gradient::polak_ribiere<double>());
    // methods.push_back(new method::conjugate_gradient::hestenes_stiefel<double>());
    methods.push_back(new method::conjugate_gradient::dai_yuan<double>());
    // methods.push_back(new method::conjugate_gradient::dai_yuan<double>());
    // methods.push_back(new method::conjugate_gradient::cg_descent<double>());
    // methods.push_back(new method::conjugate_gradient::cg_descent<double>());

    // select the corresponding line searches for each of the added methods
    map<string, double> params1;
    map<string, double> params2;
    vector<line_search::base_line_search<double>*> lsearches;
    // lsearches.push_back(new line_search::strong_wolfe<double>(params1));
    // lsearches.push_back(new line_search::strong_wolfe<double>(params1));
    // lsearches.push_back(new line_search::strong_wolfe<double>(params1));
    lsearches.push_back(new line_search::strong_wolfe<double>(params1));
    // lsearches.push_back(new line_search::approx_wolfe<double>(params2));
    // lsearches.push_back(new line_search::approx_wolfe<double>(params2));
    // lsearches.push_back(new line_search::strong_wolfe<double>(params1));

    for (int i = 0; i < methods.size(); ++i) {
        function::function<double> f = func::getFunction();

        for (int n = 10; n <= 1000; n *= 10) {
            method::base_method<double>* opt = methods[i];
            line_search::base_line_search<double>* ls = lsearches[i];

            for (int j = 0; j < t; ++j) {
                ofstream ofs(calc_filename(opt, ls, n), ofstream::app); // append to file

                la::vec<double> x = f.starting_point(n);
                //this_thread::sleep_for(chrono::milliseconds(500)); // rest before optimizing
                opt->operator()(f, *ls, x);
                ofs << opt->get_cpu_time() << "\n";

                ofs.close();
            }

            if (opt->get_gr_norm() > opt->get_epsilon()) {
                // exited on either max. iterations or work precision
                ofstream ofs(calc_filename(opt, ls, n), ofstream::app); // append to file
                ofs << "*exited on " << (opt->get_iter_count() == opt->get_max_iter() ? "max. iterations" : "work precision") << "\n";
                ofs << "*gradient norm: " << opt->get_gr_norm() << "\n";
                ofs.close();
            }
        }
    }

    return 0;

}
