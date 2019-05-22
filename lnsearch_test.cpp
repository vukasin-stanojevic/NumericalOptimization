#include <bits/stdc++.h>
#include "la.h"
#include "lnsearch.h"
#include "lautils.h"
#include "functions.h"

using namespace std;
using namespace la;

double f1(vecd x) {
	return 2*sin(x[0]) + 3*cos(x[1]) + x[0]*x[0] + x[1] * x[1];
}

vecd g1(vecd x) {
	return {2*cos(x[0]) + 2*x[0], -3*sin(x[1]) + 2*x[1]};
}

double f2(vecd x) {
	double r = 10*(x[1]-x[0]*x[0]);
	r *= r;
	return r + (1-x[0])*(1-x[0]);
}

vecd g2(vecd x) {
	return {
		2*(x[0] - 1) + 400*x[0]*(x[0]*x[0] - x[1]),
		200*(x[1] - x[0]*x[0])
	};
}

// sluzi samo za neko osnovno testiranje
template<class real, class func_t, class grad_t>
void simple_gradient_descent(
	string ls_method,
	vec<real> x0,
	func_t f,
	grad_t g,
	map<string, real> params = {})
{
	size_t steps = 0;
	while (norm(g(x0)) > 1e-8 && steps++ < 1000) {
		cerr << x0 << "   grad: ";
		cerr << g(x0) << '\n';
		auto p = -g(x0);
		x0 += p * line_search(ls_method, x0, p, f, g, params);
	}
	cerr << "steps = " << steps << '\n';
}

template<class real, class func_t, class grad_t>
void momentum(
	string ls_method,
	vec<real> x0,
	func_t f,
	grad_t g,
	map<string, real> params = {})
{
	int steps = 0;
	auto p = -g(x0);
	while (norm(g(x0)) > 1e-7 && steps++ < 1000) {
		cerr << x0 << "   gnorm = " << norm(g(x0)) << '\n';
		p = p * 0.9 - g(x0) * 0.1;
		x0 += p * line_search(ls_method, x0, p, f, g, params);
	}
	cerr << "steps = " << steps << '\n';
}

template<class real, class func_t, class grad_t>
void fletcher_reeves(
	string ls_method,
	vec<real> x0,
	func_t f,
	grad_t g,
	map<string, real> params = {})
{
	int steps = 0;
	auto p0 = -g(x0);
	real a0 = line_search(ls_method, x0, p0, f, g, params);
	auto x1 = x0 + p0*a0;
	auto s0 = p0;

	while (norm(g(x1)) > 1e-7 && steps++ < 100000) {
		cerr << x1 << "   gnorm = " << norm(g(x1)) << '\n';
		auto p1 = -g(x1); // steepest direction at xn
		auto beta1 = x1.dot(x1) / x0.dot(x0); // FR beta
		auto s1 = p1 + s0*beta1; // update the conjugate direction
		auto a1 = line_search(ls_method, x1, s1, f, g, params);

		auto x2 = x1 + s1*a1;

		x1 = x2;
		s0 = s1;
	}
}

template<class T>
struct call_count_wrapper {
	T f;
	shared_ptr<int> cc;

	call_count_wrapper(T f) : f(f), cc(make_shared<int>(0)) {}

	template<class... U>
	auto operator() (U... u) {
		++*cc;
		return f(u...);
	}

	int count() const {
		return *cc;
	}
};

template<class T>
call_count_wrapper<T> make_call_count_wrapper(T f) {
	return call_count_wrapper<T>(f);
}

int main() {
	ios_base::sync_with_stdio(false);
	cin.tie(nullptr);
	cout.tie(nullptr);
	cerr.tie(nullptr);

	string fname = "extended_psc1";

	auto f = functions::function<double>(fname);
	auto g = functions::gradient<double>(fname);
	vecd x0 = functions::starting_point<double>(fname, 4);

	auto f2 = make_call_count_wrapper(f);

	cerr.precision(8);
	cerr << fixed;
	fletcher_reeves("wolfe", x0, f2, g);
	cerr << f2.count() << '\n';
}
