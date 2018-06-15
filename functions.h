#ifndef _FUNCTIONS_H
#define _FUNCTIONS_H

#include "la.h"
#include <cmath>
#include <string>

/*
	Dostupne funkcije:
	Ime, pozicija u pdf-u sa funkcijama (naziv u programu)

	Extended PSC1, 6/3 (extended_psc1)
	Full Hessian FH2, 6/6 (full_hessian_fh2)
	Extended quadratic penalty QP2 7/6 (extended_qp2)
	Partial Perturbed Quadratic 9/-1 (pp_quad)
	EXPLIN1 11/4 (explin1)
*/

namespace functions {

template<class real>
real quadratic(const la::vec<real>& v) {
	real z = 0;
	for (real x : v)
		z += x*x;
	return z;
}

// Extended PSC1, 6/3 (extended_psc1)

template<class real>
real extended_psc1_f(const la::vec<real>& v) {
	if (v.size() % 2 || v.size() == 0)
		throw "extended_psc1: n must be even and positive";
	real z = 0;
	for (size_t i=0; i<v.size(); i+=2) {
		real t = v[i]*v[i] + v[i+1]*v[i+1] + v[i]*v[i+1];
		z += t*t;
		t = sin(v[i]);
		z += t*t;
		t = cos(v[i+1]);
		z += t*t;
	}
	return z;
}

template<class real>
la::vec<real> extended_psc1_g(const la::vec<real>& v) {
	if (v.size() % 2 || v.size() == 0)
		throw "extended_psc1: n must be even and positive";
	la::vec<real> z(v.size(), 0.0);
	for (size_t i=0; i<v.size(); i+=2) {
		real t = v[i]*v[i] + v[i+1]*v[i+1] + v[i]*v[i+1];
		z[i] += 2*t*(2*v[i] + v[i+1]);
		z[i] += 2*sin(v[i])*cos(v[i]);

		z[i+1] += 2*t*(2*v[i+1] + v[i]);
		z[i+1] -= 2*cos(v[i+1])*sin(v[i+1]);
	}
	return z;
}

template<class real>
la::vec<real> extended_psc1_x0(size_t n) {
	if (n % 2 || n == 0)
		throw "extended_psc1: n must be even and positive";
	la::vec<real> z(n, 0);
	for (size_t i=0; i<n; i+=2) {
		z[i] = 3;
		z[i+1] = 0.1;
	}
	return z;
}

// Full Hessian FH2, 6/6 (full_hessian_fh2)

template<class real>
real full_hessian_fh2_f(const la::vec<real>& v) {
	if (v.size() == 0)
		throw "full_hessian_fh2: n must be positive";
	real z = 0, ps = 0;
	z = (v[0] - 5);
	z *= z;
	ps = v[0];
	for (size_t i=1; i<v.size(); i++) {
		ps += v[i];
		z += (ps - 1) * (ps - 1);
	}
	return z;
}

template<class real>
la::vec<real> full_hessian_fh2_g(const la::vec<real>& v) {
	if (v.size() == 0)
		throw "full_hessian_fh2: n must be positive";
	// krajnji rezultat
	la::vec<real> z(v.size(), 0.0);
	// prefiksne sume
	la::vec<real> ps(v.size(), 0.0);
	ps[0] = v[0];
	for (size_t i=1; i<v.size(); i++)
		ps[i] = ps[i-1] + v[i];
	
	// pocetni rezultat
	real t = -2*(real)v.size();
	for (size_t i=0; i<v.size(); i++)
		t += (v.size() - i) * v[i] * 2;
	z[0] = t - 8;
	for (size_t i=1; i<v.size(); i++) {
		t -= 2*ps[i-1];
		t += 2;
		z[i] = t;
	}
	return z;
}

template<class real>
la::vec<real> full_hessian_fh2_x0(size_t n) {
	if (n == 0)
		throw "full_hessian_fh2: n must be positive";
	return la::vec<real>(n, 0.01);
}

// Extended quadratic penalty QP2 7/6 (extended_qp2)

template<class real>
real extended_qp2_f(const la::vec<real>& v) {
	if (v.size() == 0)
		throw "extended_qp2: n must be positive";
	real z = 0;
	for (size_t i=0; i<v.size()-1; i++) {
		real t = v[i]*v[i] - sin(v[i]);
		z += t*t;
	}
	real p = -100;
	for (size_t i=0; i<v.size(); i++)
		p += v[i]*v[i];
	z += p*p;
	return z;
}

template<class real>
la::vec<real> extended_qp2_g(const la::vec<real>& v) {
	if (v.size() == 0)
		throw "extended_qp2: n must be positive";
	la::vec<real> z(v.size(), 0.0);
	for (size_t i=0; i<v.size()-1; i++)
		z[i] += 2*(2*v[i]-cos(v[i]))*(v[i]*v[i] - sin(v[i]));
	real t = -100;
	for (size_t i=0; i<v.size(); i++)
		t += v[i]*v[i];
	for (size_t i=0; i<v.size(); i++)
		z[i] += 4*v[i]*t;
	return z;
}

template<class real>
la::vec<real> extended_qp2_x0(size_t n) {
	if (n == 0)
		throw "extended_qp2: n must be positive";
	return la::vec<real>(n, 0.5);	
}

// Partial Perturbed Quadratic 9/-1 (pp_quad)

template<class real>
real pp_quad_f(const la::vec<real>& v) {
	if (v.size() == 0)
		throw "pp_quad: n must be positive";
	real z = 0;
	z += v[0]*v[0];
	real ps = 0;
	for (size_t i=0; i<v.size(); i++) {
		ps += v[i];
		z += v[i]*v[i]*(i+1);
		z += ps*ps / 100;
	}
	return z;
}

template<class real>
la::vec<real> pp_quad_g(const la::vec<real>& v) {
	if (v.size() == 0)
		throw "pp_quad: n must be positive";
	la::vec<real> z(v.size(), 0.0);
	la::vec<real> ps(v.size());
	ps[0] = v[0];
	real t = 0;
	for (size_t i=1; i<v.size(); i++)
		ps[i] = ps[i-1] + v[i];
	for (size_t i=0; i<v.size(); i++)
		t += v[i] * (v.size() - i) * 2;

	z[0] = t / 100 + v[0] * 4;
	for (size_t i=1; i<v.size(); i++) {
		t -= 2*ps[i-1];
		z[i] = t / 100 + v[i] * (i+1) * 2;
	}
	return z;
}

template<class real>
la::vec<real> pp_quad_x0(size_t n) {
	if (n == 0)
		throw "pp_quad: n must be positive";
	return la::vec<real>(n, 0.5);
}

// EXPLIN1 11/4 (explin1)

template<class real>
real explin1_f(const la::vec<real>& v) {
	if (v.size() == 0)
		throw "explin1: n must be positive";
	real z = 0;
	for (size_t i=0; i<v.size()-1; i++)
		z += exp(0.1 * v[i] * v[i+1]);
	for (size_t i=0; i<v.size(); i++)
		z -= v[i] * 10 * (i+1);
	return z;
}

template<class real>
la::vec<real> explin1_g(const la::vec<real>& v) {
	if (v.size() == 0)
		throw "explin1: n must be positive";
	la::vec<real> z(v.size(), 0.0);
	for (size_t i=0; i<v.size()-1; i++) {
		z[i] += exp(v[i]*v[i+1] / 10) * v[i+1] / 10;
		z[i+1] += exp(v[i]*v[i+1] / 10) * v[i] / 10;
	}
	for (size_t i=0; i<v.size(); i++)
		z[i] -= (real)10*(i+1);
	return z;
}

template<class real>
la::vec<real> explin1_x0(size_t n) {
	if (n == 0)
		throw "explin1: n must be positive";
	return la::vec<real>(n, 0.0);
}

// Za izvoz!

template<class real>
auto function(std::string name) {
	if (name == "extended_psc1")
		return extended_psc1_f<real>;
	if (name == "full_hessian_fh2")
		return full_hessian_fh2_f<real>;
	if (name == "extended_qp2")
		return extended_qp2_f<real>;
	if (name == "pp_quad")
		return pp_quad_f<real>;
	if (name == "explin1")
		return explin1_f<real>;
	throw "function not implemented";
}

template<class real>
auto gradient(std::string name) {
	if (name == "extended_psc1")
		return extended_psc1_g<real>;
	if (name == "full_hessian_fh2")
		return full_hessian_fh2_g<real>;
	if (name == "extended_qp2")
		return extended_qp2_g<real>;
	if (name == "pp_quad")
		return pp_quad_g<real>;
	if (name == "explin1")
		return explin1_g<real>;
	throw "function not implemented";
}

template<class real>
auto starting_point(std::string name, size_t n) {
	if (name == "extended_psc1")
		return extended_psc1_x0<real>(n);
	if (name == "full_hessian_fh2")
		return full_hessian_fh2_x0<real>(n);
	if (name == "extended_qp2")
		return extended_qp2_x0<real>(n);
	if (name == "pp_quad")
		return pp_quad_x0<real>(n);
	if (name == "explin1")
		return explin1_x0<real>(n);
	throw "function not implemented";
}


} // namespace la

#endif