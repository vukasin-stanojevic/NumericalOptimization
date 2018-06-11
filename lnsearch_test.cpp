#include <bits/stdc++.h>
#include "la.h"
#include "lnsearch.h"
#include "lautils.h"

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
	int steps = 0;
	while (norm(g(x0)) > 1e-8 && steps++ < 1000) {
		cerr << x0 << "   grad: ";
		cerr << g(x0) << '\n';
		auto p = -g(x0);
		x0 += p * line_search(ls_method, x0, p, f, g, params);
	}
	cerr << "steps = " << steps << '\n';
}

int main() {
	ios_base::sync_with_stdio(false);
	cin.tie(nullptr);
	cout.tie(nullptr);
	cerr.tie(nullptr);

	auto f = f2;
	auto g = g2;
	vecd x0 = {0.0, 0.0};

	simple_gradient_descent("armijo", x0, f, g);
	simple_gradient_descent("wolfe", x0, f, g);
	simple_gradient_descent("strong_wolfe", x0, f, g);
	simple_gradient_descent("goldstein", x0, f, g,
		{{"steepness", 0.1}});
	simple_gradient_descent("fixed_line_search", x0, f, g,
		{{"initial_step", 0.001}});
}