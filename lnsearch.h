#include <map>
#include "la.h"
#include "lautils.h"

#include <iostream>
using namespace std;

template<class real, class func_t, class grad_t>
real armijo(
	la::vec<real> x0,
	la::vec<real> d,
	func_t f,
	grad_t g,
	real steepness,
	real initial_step)
{
	real f0 = f(x0); // vrednost u polaznoj tacki

	real pad = g(x0).dot(d); // kojom brzinom raste f u pravcu d
	// drugim recima, to je priblizna vrednost f(x0+d) - f(x0)

	real a_curr = initial_step; // nalazimo se u tacki x0 + d*a

	real f_curr, f_prev, a_prev;
	f_curr = f(x0 + d * a_curr);

	size_t steps = 1;

	while (f_curr > f0 - steepness * a_curr * pad) {
		real a_new;
		if (steps == 1) {
			// nadji sledecu tacku kvadratnom interpolacijom
			a_new = pad*a_curr*a_curr / 2 / (f0 - f_curr + pad*a_curr);
		} else {
			// nadji kubnom interpolacijom
			real cubic = a_prev * a_prev * (f_curr - f0);
			cubic -= a_curr * a_prev * a_prev * pad;
			cubic += a_curr * a_curr * (f0 - f_prev + a_prev * pad);
			cubic /= a_curr*a_curr*(a_curr - a_prev)*a_prev*a_prev;

			real quadr = -cubic*a_curr*a_curr*a_curr - f0 + f_curr - a_curr*pad;
			quadr /= a_curr * a_curr;

			a_new = (-quadr+sqrt(quadr*quadr - 3*cubic*pad)) / (3 * cubic);
		}

		a_prev = a_curr;
		a_curr = a_new;

		f_prev = f_curr;
		f_curr = f(x0 + d * a_curr);

		++steps;
	}

	return a_curr;
}

template<class real, class func_t, class grad_t>
real line_search(
	const std::string& method_name,
	la::vec<real> x0,
	la::vec<real> d,
	func_t f,
	grad_t g,
	const std::map<std::string, real>& params)
{
	std::map<std::string, real> p;

	auto rest = [&]() {
		for (auto e : params)
			p[e.first] = e.second;
	};

	if (method_name == "armijo") {
		p["steepness"] = 0.5;
		p["initial_step"] = 1;
		rest();
		return armijo(x0, d, f, g, p["steepness"], p["initial_step"]);
	}

	return 0.0;
}

