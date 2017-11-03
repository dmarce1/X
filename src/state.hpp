/*
 * state.hpp
 *
 *  Created on: Nov 2, 2017
 *      Author: dmarce1
 */

#ifndef STATE_HPP_
#define STATE_HPP_

#include "defs.hpp"
static constexpr int NS = 0;
static constexpr int NF = 6 + NS;
#define FGAMMA  (5.0 / 3.0)
#define rho_i 0
#define egas_i 1
#define tau_i 2
#define momx_i  3
#define momy_i 4
#define momz_i  5

#ifdef __CUDA_ARCH__
#undef USE_CPU
#define USE_GPU
#else
typedef real simd __attribute__ ((vector_size (32)));
#endif
constexpr int simd_size = 32 / sizeof(real);

template<class T>
struct constants;

#include "simd.hpp"

template<class T = double>
class state_var {
private:
	T data[NF];
public:
	EXPORT_GLOBAL T& rho() {
		return data[0];
	}
	EXPORT_GLOBAL T& egas() {
		return data[1];
	}
	EXPORT_GLOBAL T& tau() {
		return data[2];
	}
	EXPORT_GLOBAL const T& rho() const {
		return data[0];
	}
	EXPORT_GLOBAL const T& egas() const {
		return data[1];
	}
	EXPORT_GLOBAL const T& tau() const {
		return data[2];
	}
	static std::string name(int f) {
		std::string n;
		if (f == 0) {
			n = "rho";
		} else if (f == 1) {
			n = "egas";
		} else if (f == 2) {
			n = "tau";
		} else if (f > 2 && f < 5) {
			n = std::string("mom_") + char('x' + f - 2);
		} else {
			n = std::string("frac_") + std::to_string(f - 5);
		}
		return n;
	}
	EXPORT_GLOBAL T& mom(int d) {
		return data[3 + d];
	}
	EXPORT_GLOBAL const T& mom(int d) const {
		return data[3 + d];
	}
	EXPORT_GLOBAL T& scalar(int d) {
		return data[NF - NS + d];
	}
	EXPORT_GLOBAL const T& scalar(int d) const {
		return data[NF - NS + d];
	}
	EXPORT_GLOBAL T& operator[](int d) {
		return data[d];
	}
	EXPORT_GLOBAL const T& operator[](int d) const {
		return data[d];
	}
	EXPORT_GLOBAL T ekinetic() const {
		T ek = constants<T>::zero;
		for (int dim = 0; dim != NDIM; ++dim) {
			ek += 0.5 * mom(dim) * mom(dim) / rho();
		}
		return ek;
	}
	EXPORT_GLOBAL T eint() const {
		T ei = egas() - ekinetic();
		ei += T(ei < 0.01 * egas()) * (pow(tau(), FGAMMA) - ei);
		return ei;
	}
	EXPORT_GLOBAL T pressure() const {
		return (FGAMMA - 1.0) * eint();
	}
	EXPORT_GLOBAL T sound_speed() const {
		return sqrt(FGAMMA * pressure() / rho());
	}
	EXPORT_GLOBAL T vel(int dim) const {
		return mom(dim) / rho();
	}
	EXPORT_GLOBAL T max_lambda() const {
		T vmax = fabs(vel(0));
		for (int d = 1; d < NDIM; ++d) {
			vmax = fmax(vmax, fabs(vel(d)));
		}
		return vmax + sound_speed();
	}
	EXPORT_GLOBAL state_var<T>& zero() {
		for (int f = 0; f != NF; ++f) {
			(*this)[f] = 0.0;
		}
		return *this;
	}
	EXPORT_GLOBAL state_var<T> operator-(const state_var<T>& other) const {
		state_var<T> result;
		for (int f = 0; f != NF; ++f) {
			result[f] = other[f] - (*this)[f];
		}
		return result;
	}
	EXPORT_GLOBAL void set_tau() {
		tau() = pow(eint(), T(1) / FGAMMA);
	}
	state_var() = default;
	state_var(const state_var&) = default;
	state_var(state_var&&) = default;
	state_var& operator=(const state_var&) = default;
	state_var& operator=(state_var&&) = default;

	EXPORT_GLOBAL state_var(real* u, int i, int D) {
		for (int f = 0; f != NF; ++f) {
			(*this)[f] = u[f * D + i];
		}
	}
};

EXPORT_GLOBAL inline void set_state_var(real* u, int i, state_var<real>& var, int D) {
	for (int f = 0; f != NF; ++f) {
		u[f * D + i] = var[f];
	}
}

template<class T>
struct state_pair {
	state_var<T> L;
	state_var<T> R;
public:
	EXPORT_GLOBAL state_var<T> flux(int dim, T& a) const {
		state_var<T> F;
		const T pr = R.pressure();
		const T pl = L.pressure();
		const T rhorinv = 1.0 / R.rho();
		const T rholinv = 1.0 / L.rho();
		const T cr = sqrt(FGAMMA * pr * rhorinv);
		const T cl = sqrt(FGAMMA * pl * rholinv);
		const T vr = R.mom(dim) * rhorinv;
		const T vl = L.mom(dim) * rholinv;
		const T ar = cr + abs(vr);
		const T al = cl + abs(vl);
		a = max(ar, al);
		for (int f = 0; f != NF; ++f) {
			F[f] = (vr * R[f] + vl * L[f] - a * (R[f] - L[f])) * 0.5;
		}
		F.egas() += (pr * vr + pl * vl) * 0.5;
		F.mom(dim) += (pr + pl) * 0.5;
		return F;
	}
};

#endif /* STATE_HPP_ */
