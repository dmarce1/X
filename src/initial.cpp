#include "initial.hpp"

state_var<real> initial_value(initial_value_type type, real x, real y, real z) {
	state_var<real> u;
	u.zero();
	switch (type) {
	case SOD:
		if (x < 0.5) {
			u.rho() = 1.0;
			u.egas() = 2.5;
		} else {
			u.rho() = 0.125;
			u.egas() = 0.25;
		}
		break;
	}
	u.set_tau();
	return u;
}

real problem_gamma(initial_value_type type) {
	real fgamma = 5.0 / 3.0;
	switch (type) {
	case SOD:
		fgamma = 7.0 / 5.0;
		break;
	}
	return fgamma;
}
