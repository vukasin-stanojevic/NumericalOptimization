#ifndef APPROX_WOLFE_H_INCLUDED
#define APPROX_WOLFE_H_INCLUDED

#include <cmath>
#include "base_line_search.h"

namespace opt {
namespace line_search {

template<class real>
class approx_wolfe : public base_line_search<real> {
private:
    real steepness; // rho; delta in paper
    real initial_step; // start point
    real sigma;
    real theta; // used in the update rule
    real gamma; // determines when a bisection step is performed
    real eps; // error tolerance
    real prev_step; // previous line search step size
public:
    approx_wolfe(std::map<std::string, real>& params) {
        std::map<std::string, real> p;
        p["steepness"] = 0.1;
        p["initial_step"] = 1;
        p["sigma"] = 0.9;
        p["theta"] = 0.5;
        p["gamma"] = 0.66;
        p["eps"] = 1e-6;
        this->rest(p, params);
        steepness = p["steepness"];
        initial_step = p["initial_step"];
        sigma = p["sigma"];
        theta = p["theta"];
        gamma = p["gamma"];
        eps = p["eps"];
        params = p;
        prev_step = initial_step;
    }

    real operator()(function::function<real>& f, la::vec<real>& x, la::vec<real>& d) {
        this->iter_count = 0;

        int k = this->f_values.size();
        if (k == 1) {
            prev_step = initial_step;
        }

        if (this->c == 0) {
            this->c = this->f_values.end()[-1];
        }
        real epsilon = eps * this->c;

        la::vec<real>& gr = this->current_g_val;
        real phi0 = this->current_f_val;
        real der_phi0 = gr.dot(d); // derivative of Phi(t) in point x
        
        real c; // initial() out parameter
        real phi_c; // initial() out parameter

        initial(f, x, phi0, gr, der_phi0, d, k, prev_step, c, phi_c);
        
        la::vec<real> grad_c = f.gradient(x + d * c);
        real der_phi_c = grad_c.dot(d); // derivative of Phi(c) in point x

        real aj; // bracket() out parameter
        real bj; // bracket() out parameter
        real val_aj; // bracket() out parameter
        real val_bj; // bracket() out parameter
        real der_aj; // bracket() out parameter
        real der_bj; // bracket() out parameter
        
        bracket(c, phi0, phi_c, der_phi0, der_phi_c, f, x, d, 5, epsilon, aj, bj, val_aj, der_aj, val_bj, der_bj);

        real phi2;
        real der_phi2;

        while (1) {
            ++this->iter_count;

            phi2 = phi_c;
            der_phi2 = der_phi_c;

            if ((steepness*der_phi0*c >= (phi2-phi0) && der_phi2 >= sigma*der_phi0)
                || (((2*steepness-1)*der_phi0 >= der_phi2 && der_phi2 >= sigma*der_phi0) || phi2 <= phi0+epsilon)) {
                // found step size c
                this->current_f_val = phi2;
                this->current_g_val = grad_c;
                this->prev_step = c;
                return c;
            }

            real a; // secant2() out parameter
            real b; // secant2() out parameter
            real val_a; // secant2() out parameter
            real der_a; // secant2() out parameter
            real val_b; // secant2() out parameter
            real der_b; // secant2() out parameter

            secant2(aj, bj, val_aj, der_aj, val_bj, der_bj, f, phi0, x, d, epsilon, a, b, val_a, der_a, val_b, der_b);
                
            if (b - a > gamma * (bj - aj)) {
                c = (a + b) / 2.0;
                // all after eps are out parameters
                update(a, b, c, val_a, der_a, val_b, der_b, phi0, f, x, d, epsilon, a, b, val_a, der_a, val_b, der_b, phi_c, der_phi_c, grad_c);
            }
                
            aj = a;
            bj = b;
            val_aj = val_a;
            der_aj = der_a;
            val_bj = val_b;
            der_bj = der_b;
        }
    }
private:
    void initial(function::function<real>& f, la::vec<real>& x, real phi0, la::vec<real>& der0, real der_phi0, la::vec<real>& dir, size_t k, real c_old, real& c, real& phi_c) {
        real psi0 = 0.01;
        real psi1 = 0.1;
        real psi2 = 2;

        // I0 condition
        if (k == 1) {
            bool nonZeroX = true;
            for (size_t i = 0; i < x.size() && nonZeroX; ++i) {
                if (x[i] == 0.0) {
                    nonZeroX = false;
                }
            }
            
            if (nonZeroX) {
                // I0.0
                real res1 = fabs(x[0]);
                for (size_t i = 1; i < x.size(); ++i) {
                    res1 = fmax(res1, fabs(x[i]));
                }
                real res2 = fabs(der0[0]);
                for (size_t i = 1; i < der0.size(); ++i) {
                    res2 = fmax(res2, fabs(der0[i]));
                }
                c = psi0 * res1 / res2;
                phi_c = f(x + dir * c);
                return;
            }

            if (phi0 != 0.0) {
                // I0.1
                real norm = la::norm(der0);
                c = psi0 * fabs(phi0) / (norm*norm);
                phi_c = f(x + dir * c);
                return;
            }
            // I0.2
            c = 1.0;
            phi_c = f(x + dir * c);
            return;
        }

        // I1 condition
        real r = psi1 * c_old;
        real phi_r = f(x + dir * r);

        // Check whether interpolation function is convex
        if (phi0 - phi_r + r * der_phi0 < 0) {
            // Computes minimum of interpolation function that matches phi0, der_phi0, phi_r, r
            real q = 0.5*r*r*der_phi0 / (phi0 - phi_r + r * der_phi0);
            real phi_q = f(x + dir * q);

            if (phi_q < phi0) {
                c = q;
                phi_c = phi_q;
                return;
            }
        }

        // I2 condition
        c = psi2 * c_old;
        phi_c = f(x + dir * c);
    }

    void bracket(real c, real phi0, real phi_c, real der_phi0, real der_phi_c, function::function<real>& f, la::vec<real>& x, la::vec<real>& dir, real range_expansion, real eps, real& a_, real& b_, real& val_a_, real& der_a_, real& val_b_, real& der_b_) {
        real cj = c;
        real ci = 0;

        real phi_j = phi_c;
        real phi_i = phi0;

        real der_j = der_phi_c;
        real der_i = der_phi0;

        while (1) {
            if (phi_j <= phi0 + eps) {
                ci = cj;
                phi_i = phi_j;
                der_i = der_j;
            }

            if (der_j >= 0) {
                a_ = ci;
                b_ = cj;
                val_a_ = phi_i;
                der_a_ = der_i;
                val_b_ = phi_j;
                der_b_ = der_j;
                return;
            }

            if (der_j < 0 && phi_j > phi0 + eps) {
                update3(0, cj, phi0, der_phi0, phi_j, der_j, phi0, f, x, dir, eps, a_, b_, val_a_, der_a_, val_b_, der_b_);
                return;
            }

            cj *= range_expansion;
            la::vec<real> xx = x + dir * cj;
            phi_j = f(xx);
            der_j = f.gradient(xx).dot(dir);
        }
    }

    real secant(real a, real b, real der_a, real der_b) {
        real d = fmax(der_b - der_a, 1e-16);
        return (a * der_b - b * der_a) / d;
    }

    void secant2(real a, real b, real val_a, real der_a, real val_b, real der_b, function::function<real>& f, real phi0, la::vec<real>& x, la::vec<real>& dir, real eps, real& a_, real& b_, real& val_a_, real& der_a_, real& val_b_, real& der_b_) {
        real c = secant(a, b, der_a, der_b);

        real t1, t2;
        la::vec<real> t3;

        update(a, b, c, val_a, der_a, val_b, der_b, phi0, f, x, dir, eps, a_, b_, val_a_, der_a_, val_b_, der_b_, t1, t2, t3);

        if (c == a_ || c == b_) {
            real s, der_s;
            real ss, der_ss;

            if (c == b_) {
                s = b;
                ss = b_;
                der_s = der_b;
                der_ss = der_b_;
            } else {
                s = a;
                ss = a_;
                der_s = der_a;
                der_ss = der_a_;
            }

            real c_ = secant(s, ss, der_s, der_ss);
            update(a_, b_, c_, val_a_, der_a_, val_b_, der_b_, phi0, f, x, dir, eps, a_, b_, val_a_, der_a_, val_b_, der_b_, t1, t2, t3);
        }
    }

    void update(real a, real b, real c, real val_a, real der_a, real val_b, real der_b,
                real phi0, function::function<real>& f, la::vec<real>& x, la::vec<real>& dir, real eps,
                real& a_, real& b_, real& val_a_, real& der_a_, real& val_b_, real& der_b_, real& phi_c, real& der_phi_c, la::vec<real>& grad_c) {
        la::vec<real> xx = x + dir * c;
        phi_c = f(xx);
        grad_c = f.gradient(xx);
        der_phi_c = grad_c.dot(dir);

        // U0
        if (c <= a || c >= b) {
            a_ = a;
            b_ = b;
            val_a_ = val_a;
            der_a_ = der_a;
            val_b_ = val_b;
            der_b_ = der_b;
            return;
        }

        // U1
        if (der_phi_c >= 0) {
            a_ = a;
            b_ = c;
            val_a_ = val_a;
            der_a_ = der_a;
            val_b_ = phi_c;
            der_b_ = der_phi_c;
            return;
        }

        // U2
        if (der_phi_c < 0 && phi_c <= phi0 + eps) {
            a_ = c;
            b_ = b;
            val_a_ = phi_c;
            der_a_ = der_phi_c;
            val_b_ = val_b;
            der_b_ = der_b;
            return;
        }

        // U3
        if (der_phi_c < 0 && phi_c > phi0 + eps) {
            update3(a, c, val_a, der_a, phi_c, der_phi_c, phi0, f, x, dir, eps, a_, b_, val_a_, der_a_, val_b_, der_b_);
        }
    }

    void update3(real a, real b, real val_a, real der_a, real val_b, real der_b, real phi0, function::function<real>& f, la::vec<real>& x, la::vec<real>& dir, real eps, real& a_, real& b_, real& val_a_, real& der_a_, real& val_b_, real& der_b_) {
        a_ = a;
        b_ = b;

        while (1) {
            real d = (1 - theta) * a_ + theta * b_;
            la::vec<real> xx = x + dir * d;
            real phi_d = f(xx);
            real der_phi_d = f.gradient(xx).dot(dir);

            // U3a
            if (der_phi_d >= 0) {
                b_ = d;
                val_a_ = val_a;
                der_a_ = der_a;
                val_b_ = phi_d;
                der_b_ = der_phi_d;
                return;
            }

            // U3b
            if (der_phi_d < 0 && phi_d <= phi0 + eps) {
                a_ = d;
                val_a_ = phi_d;
                der_a_ = der_phi_d;
                val_b_ = val_b;
                der_b_ = der_b;
                return;
            }

            // U3c
            if (der_phi_d < 0 && phi_d > phi0 + eps) {
                b_ = d;
                val_a_ = val_a;
                der_a_ = der_a;
                val_b_ = phi_d;
                der_b_ = der_phi_d;
                return;
            }
        }
    }
};

}
}

#endif //APPROX_WOLFE_H_INCLUDED
