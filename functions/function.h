#ifndef PROJEKATC___FUNCTION_H
#define PROJEKATC___FUNCTION_H

#include "../utilities/linear_algebra.h"
#include <future>

namespace opt {
namespace function {

template<typename T>
struct reciprocal_of_sqrt_of_sum {
    T operator ()(T t1, T t2 ) const { return ((T)1)/sqrt(t1 + t2); };
};

template<typename T>
struct sqrt_of_sum {
    T operator ()(T t1, T t2 ) const { return ((T)1)/sqrt(t1 + t2); };
};

template<class real>
class function {

public:
	using func = real(*)(const la::vec<real>&);
	using grad = la::vec<real>(*)(const la::vec<real>&);
	using hess = la::mat<real>(*)(const la::vec<real>&);
	using start = la::vec<real>(*)(const size_t);

    using func_calc_job = void(*)(const la::vec<real>*, std::promise<real>&&, const size_t , const size_t);
    using grad_calc_job = void(*)(const la::vec<real>*, la::vec<real>*, const size_t, const size_t);
    using hess_calc_job = void(*)(const la::vec<real>*, la::mat<real>*, const size_t, const size_t);

	function(func f, grad g, hess h, start s) : f(f), g(g), h(h), s(s), call_count(0), grad_count(0), hess_count(0) {}

	real operator()(const la::vec<real>& x) {
		++call_count;
		return f(x);
	}

	la::vec<real> gradient(const la::vec<real>& x) {
		++grad_count;
		return g(x);
	}

	la::mat<real> hessian(const la::vec<real>& x) {
		++hess_count;
		return h(x);
	}

	la::vec<real> starting_point(const size_t n) {
		return s(n);
	}

	size_t get_call_count() const {
		return call_count;
	}

	size_t get_grad_count() const {
		return grad_count;
	}

	size_t get_hess_count() const {
		return hess_count;
	}

    static void calculate_hessian_multithread(const la::vec<real>* x, la::mat<real>* hess, hess_calc_job h) {
        unsigned int processor_count = std::thread::hardware_concurrency();
        processor_count = processor_count > la::MAX_THREAD_NUM ? la::MAX_THREAD_NUM : processor_count;
        if (processor_count % 2 == 1) {
            processor_count--;
        }

        if (processor_count > 1) {
            std::vector<std::thread> threads;

            size_t work_by_thread = hess->rows() / processor_count;
            int last_thread_additional_work = hess->rows() - work_by_thread * processor_count;

            int k;
            for (k = 0; k < processor_count - 1; k++) {
                threads.push_back(std::thread(h, x, hess, k*work_by_thread, (k+1)*work_by_thread));
            }
            threads.push_back(std::thread(h, x, hess, k*work_by_thread, (k+1)*work_by_thread + last_thread_additional_work));

            for (int i = 0; i < threads.size(); ++i) {
                threads[i].join();
            }
        } else {
            h(x, hess, 0, hess->rows());
        }
    }

	static void calculate_gradient_multithread(const la::vec<real>* x, la::vec<real>* grad, grad_calc_job g) {
        unsigned int processor_count = std::thread::hardware_concurrency();
        processor_count = processor_count > la::MAX_THREAD_NUM ? la::MAX_THREAD_NUM : processor_count;
        if (processor_count % 2 == 1) {
            processor_count--;
        }

        if (processor_count > 1) {
            std::vector<std::thread> threads;

            size_t work_by_thread = x->size() / processor_count;
            int last_thread_additional_work = x->size() - work_by_thread * processor_count;

            int k;
            for (k = 0; k < processor_count - 1; k++) {
                threads.push_back(std::thread(g, x, grad, k*work_by_thread, (k+1)*work_by_thread));
            }
            threads.push_back(std::thread(g, x, grad, k*work_by_thread, (k+1)*work_by_thread + last_thread_additional_work));

            for (int i = 0; i < threads.size(); ++i) {
                threads[i].join();
            }
        } else {
            g(x, grad, 0, x->size());
        }
	}

	static real calculate_value_multithread(const la::vec<real>* x, func_calc_job f) {
        unsigned int processor_count = std::thread::hardware_concurrency();
        processor_count = processor_count > la::MAX_THREAD_NUM ? la::MAX_THREAD_NUM : processor_count;
        real val = 0;

        if (processor_count > 1) {
            std::vector<std::thread> threads;
            std::vector<std::future<real>> futures;

            size_t work_by_thread = x->size() / processor_count;
            int last_thread_additional_work = x->size() - work_by_thread * processor_count;

            int k;
            for (k = 0; k < processor_count - 1; k++) {
                std::promise<real> p;
                std::future<real> fut = p.get_future();
                futures.push_back(std::move(fut));
                threads.push_back(std::thread(f, x, std::move(p), k*work_by_thread, (k+1)*work_by_thread));
            }
            std::promise<real> p;
            std::future<real> fut = p.get_future();
            futures.push_back(std::move(fut));
            threads.push_back(std::thread(f, x, std::move(p), k*work_by_thread, (k+1)*work_by_thread + last_thread_additional_work));

            for (int i = 0; i < futures.size(); ++i) {
                threads[i].join();
                val += futures[i].get();
            }
        } else {
            std::promise<real> p;
            std::future<real> fut = p.get_future();
            f(x, std::move(p), 0, x->size());
            val = fut.get();
        }
        return val;
	}
protected:
	func f;
	grad g;
	hess h;
	start s;
	size_t call_count;
	size_t grad_count;
	size_t hess_count;
};

}
}

#endif //PROJEKATC___FUNCTION_H
