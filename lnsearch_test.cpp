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

// sluzi samo za neko osnovno testiranje
template<class real, class func_t, class grad_t>
void simple_gradient_descent(
	string ls_method,
	vec<real> x0,
	func_t f,
	grad_t g)
{
	int steps = 0;
	while (norm(g(x0)) > 1e-8 && steps++ < 100) {
		cerr << x0 << "   grad: ";
		cerr << g(x0) << '\n';
		auto p = g(x0) * -1;
		x0 += p * line_search(ls_method, x0, p, f, g, {});
	}
	cerr << "steps = " << steps << '\n';
}

int main() {
	ios_base::sync_with_stdio(false);
	cin.tie(nullptr);
	cout.tie(nullptr);
	cerr.tie(nullptr);

	simple_gradient_descent("wolfe", vecd{1.0, 1.0}, f1, g1);
	simple_gradient_descent("armijo", vecd{1.0, 1.0}, f1, g1);
}