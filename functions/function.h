#ifndef PROJEKATC___FUNCTION_H
#define PROJEKATC___FUNCTION_H

#include "../linear_algebra.h"

namespace opt {
namespace function {

template<class real>
class function {
public:
	using func = real(*)(const la::vec<real>&);
	using grad = la::vec<real>(*)(const la::vec<real>&);
	using hess = la::mat<real>(*)(const la::vec<real>&);
	using start = la::vec<real>(*)(const size_t);

	function(func f,grad g, hess h, start s) : f(f), g(g), h(h), s(s), call_count(0), grad_count(0), hess_count(0) {}

	real operator()(const la::vec<real>& v) {
		++call_count;
		return f(v);
	}

	la::vec<real> gradient(const la::vec<real> &v) {
		++grad_count;
		return g(v);
	}

	la::mat<real> hessian(const la::vec<real> &v) {
		++hess_count;
		return h(v);
	}

	la::vec<real> starting_point(const size_t n) {
		return s(n);
	}

	size_t get_call_count(){
		return call_count;
	}

	size_t get_grad_count(){
		return grad_count;
	}

	size_t get_hess_count(){
		return hess_count;
	}
private:
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
