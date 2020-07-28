//
// Created by vukasin on 7/28/20.
//

#ifndef NUMERICALOPTIMIZATION_TEST_MULTITHREAD_H
#define NUMERICALOPTIMIZATION_TEST_MULTITHREAD_H

#include <iostream>
#include <fstream>
#include "methods/base_method.h"
#include "functions/function.h"
#include "line_searches/base_line_search.h"
using namespace std;

template<typename T>
void test(opt::method::base_method<T>* method, opt::function::function<T>& f, opt::line_search::base_line_search<T>& ls, int number_of_runs = 25) {
    //std::cout.width (10);
    ofstream ofs(method->get_method_name(), ofstream::out);
    ofs.width(15);
    unsigned int processor_count = std::thread::hardware_concurrency();
    int n = 100;
    int old_val = la::MAX_THREAD_NUM;
//    cout.precision(10);
//    cout << fixed;

    ofs << "Method: " << method->get_method_name() << endl;
    ofs << "Max. thread number: " << processor_count << endl;
    ofs << "Average measures based on: " << number_of_runs << " runs" << endl;
    ofs << "__________________________________________________________" << endl << endl;
    while(n <= 100000) {
        ofs << "Dimensionality: " << n << endl;

        double s = 0.0;
        double prev_time;
        double new_time, first_time;
        double improvement;
        for (int t_num = 1; t_num <= processor_count; t_num++) {
            la::MAX_THREAD_NUM = t_num;
            for (int run = 0; run < number_of_runs; run++) {
                auto x = f.starting_point(n);
                method->operator()(f, ls, x);
                s += method->get_cpu_time();
            }
            new_time = s / number_of_runs;
            if (t_num == 1) {
                improvement = 0;
                first_time = new_time;
            } else {
                improvement = -100*(new_time - prev_time)/prev_time;
            }
            prev_time = new_time;
            ofs << "Number of threads: " << t_num << ",\taverage time: " << new_time << "s,\timprovement to prev: " << improvement << "%\timprovement to one thread: " << -100*(new_time - first_time)/first_time<<"%\n";
            s = 0.0;
        }
        ofs << endl;
        n *= 10;
    }

    ofs.close();
    la::MAX_THREAD_NUM = old_val;
}


#endif //NUMERICALOPTIMIZATION_TEST_MULTITHREAD_H
