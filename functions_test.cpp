#include <bits/stdc++.h>
#include "functions.h"
using namespace std;

int main() {
	ios_base::sync_with_stdio(false);
	cin.tie(nullptr);
	cout.tie(nullptr);
	cerr.tie(nullptr);

	auto f = functions::function<double>("extended_psc1");
	auto g = functions::gradient<double>("extended_psc1");
	auto x0 = functions::starting_point<double>("extended_psc1", 4);

	cerr << f({1, 2, 3, 4}) << '\n';
	cerr << g({1, 2, 3, 4}) << '\n';
	cerr << x0 << '\n';

	f = functions::function<double>("full_hessian_fh2");
	g = functions::gradient<double>("full_hessian_fh2");
	x0 = functions::starting_point<double>("full_hessian_fh2", 5);

	cerr << f({1, 2, 7, 3, 11}) << '\n';
	cerr << g({1, 2, 7, 3, 11}) << '\n';
	cerr << x0 << '\n';

	f = functions::function<double>("extended_qp2");
	g = functions::gradient<double>("extended_qp2");
	x0 = functions::starting_point<double>("extended_qp2", 3);

	cerr << f({1, 2, 5}) << '\n';
	cerr << g({1, 2, 5}) << '\n';
	cerr << x0 << '\n';

	f = functions::function<double>("pp_quad");
	g = functions::gradient<double>("pp_quad");
	x0 = functions::starting_point<double>("pp_quad", 4);

	cerr << f({1, 7, 2, 19}) << '\n';
	cerr << g({1, 7, 2, 19}) << '\n';
	cerr << x0 << '\n';

	f = functions::function<double>("explin1");
	g = functions::gradient<double>("explin1");
	x0 = functions::starting_point<double>("explin1", 4);

	cerr << f({0.1, 0.7, 0.34, 0.99}) << '\n';
	cerr << g({0.1, 0.7, 0.34, 0.99}) << '\n';
	cerr << x0 << '\n';
}