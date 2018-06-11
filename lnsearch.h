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
real wolfe(
	la::vec<real> x0,
	la::vec<real> d,
	func_t f,
	grad_t g,
	real steepness,
	real initial_step,
	real sigma,
	real xi,
	real max_step,
	real step_factor)
{
	real a1 = 0, a2 = initial_step;
	real f0 = f(x0);
	real f1 = f0;
	real f2 = f(x0 + d*a2);
	real pad0 = g(x0).dot(d);
	real pad1 = pad0;
	real pad2 = g(x0 + d*a2).dot(d);

	size_t steps = 1;

	// low effort prepisivanje ali znam sta radi
	auto interp = [](real t1, real t2, real val1, real val2, real der1, real der2)
	{
		real d1 = der1+der2-3*(val1-val2)/(t1-t2);
		real d2 = sqrt(d1*d1-der1*der2);
		real tmp = t2 - (t2 - t1)*(der2 + d2 - d1)/(der2 - der1 + 2*d2);
		real t;

		if (tmp >= 0) {
			d2 = sqrt(d1*d1-der1*der2);
			t = t2 - (t2 - t1)*(der2 + d2 - d1)/(der2 - der1 + 2*d2);
		} else {
			t = t1 - 1;
		}

		if (t < t1 || t > t2)
			t = t1;
		return t;
	};

	auto noc_zoom = [&]() {
		double a = 0.0;
		while (1) {
			if (a1 < a2)
				a = interp(a1, a2, f1, f2, pad1, pad2);
			else
				a = interp(a2, a1, f2, f1, pad2, pad1);

			real ff = f(x0 + d*a);
			real pad = g(x0 + d*a).dot(d);

			if ((abs(ff - f1) / (1 + abs(ff)) < xi) ||
				(abs(ff - f2) / (1 + abs(ff)) < xi))
			{
				return a;
			}

			if ((ff > f0 + steepness*a*pad0) || (ff >= f1)) {
				a2 = a;
				f2 = ff;
				pad2 = pad;
			} else {
				if (pad1 >= sigma * pad0)
					return a;
				a1 = a;
				f1 = ff;
				pad1 = pad;
			}
		}
		return a;
	};

	while (1) {
		// <----- armijo uslov ----->
		if (f2 > f0 + pad0*steepness*a2 || (f1 <= f2 && steps > 1))
			return noc_zoom();

		if (pad2 >= sigma*pad1)
			return a2;

		a1 = a2;
		f1 = f2;
		pad1 = pad2;

		a2 = min(max_step, a2*step_factor);
		f2 = f(x0 + d*a2);
		pad2 = g(x0 + d*a2).dot(d);
		++steps;
	}
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
		p["steepness"] = 1e-4;
		p["initial_step"] = 1;
		rest();
		return armijo(x0, d, f, g, p["steepness"], p["initial_step"]);
	}

	if (method_name == "wolfe") {
		p["steepness"] = 1e-4;
		p["initial_step"] = 1;
		p["sigma"] = 0.9;
		p["xi"] = 1e-3;
		p["max_step"] = 1e10;
		p["step_factor"] = 10;
		rest();

		return wolfe(x0, d, f, g,
			p["steepness"],
			p["initial_step"],
			p["sigma"],
			p["xi"],
			p["max_step"],
			p["step_factor"]);
	}

	throw "unknown method name";
	return 0.0;
}

