#ifndef PROJEKATC___FUNCTION_H
#define PROJEKATC___FUNCTION_H

#include "../utilities/linear_algebra.h"

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
