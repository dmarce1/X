/*
 * hydro.hpp
 *
 *  Created on: Nov 3, 2017
 *      Author: dmarce1
 */

#ifndef HYDRO_HPP_
#define HYDRO_HPP_

#include "defs.hpp"
#include "state.hpp"

#ifdef __CUDA_ARCH__
inline EXPORT_GLOBAL
real minmod(const real& a, const real& b) {
	const real sgn = copysign(half, a) + copysign(half, b);
	return sgn * fmin(fabs(a), fabs(b));
}
#endif

template<class T>
EXPORT_GLOBAL
void hydro_compute_u(real* U, state_var<T>* dU, int xi, int yi, int zi, int N[NDIM], T dt, int rk) {
	const int nx = N[0];
	const int ny = N[1];
	const int nz = N[2];
	const int sz = nx * ny * nz;
	const int idx = xi + nx * (yi + ny * zi);
	for (int f = 0; f != NF; ++f) {
		U[f * sz + idx] += dt * dU[idx][f];
	}
	if (rk == 1) {
		state_var<real> u00(U, idx, nx * ny * nz);
		T max_egas = u00.egas();
		const int idxxp = (xi + 1) + nx * (yi + ny * zi);
		const int idxxm = (xi - 1) + nx * (yi + ny * zi);
		const int idxyp = xi + nx * ((yi + 1) + ny * zi);
		const int idxym = xi + nx * ((yi - 1) + ny * zi);
		const int idxzp = xi + nx * (yi + ny * (zi + 1));
		const int idxzm = xi + nx * (yi + ny * (zi - 1));
		max_egas = fmax(max_egas, U[egas_i * sz + idxxp]);
		max_egas = fmax(max_egas, U[egas_i * sz + idxxm]);
		max_egas = fmax(max_egas, U[egas_i * sz + idxyp]);
		max_egas = fmax(max_egas, U[egas_i * sz + idxym]);
		max_egas = fmax(max_egas, U[egas_i * sz + idxzp]);
		max_egas = fmax(max_egas, U[egas_i * sz + idxzm]);
		const T eint = u00.egas() - u00.ekinetic();
		T sw = T(eint > T(0.1) * max_egas);
		u00.tau() += sw * (pow(eint, 1.0 / FGAMMA) - u00.tau());
		U[tau_i * sz + idx] = u00.tau();
	}
}

template<class T>
EXPORT_GLOBAL
T hydro_compute_du(const state_var<T>& u00, const state_var<T>& um2, const state_var<T>& um1, const state_var<T>& up1, const state_var<T>& up2,
		state_var<T>& dU, real dx[NDIM], int rk, int dim) {
	T ap, am, a;
	real beta;
	if (rk == 1) {
		{
			if (dim == 0)
				for (int f = 0; f != NF; ++f) {
					dU[f] = -half * dU[f];
				}
		}
		beta = half;
	} else if (rk == 0) {
		if (dim == 0) {
			for (int f = 0; f != NF; ++f) {
				dU[f] = constants<T>::zero;
			}
		}
		beta = one;
	}
	const real c0 = beta / dx[dim];
	state_pair<T> um, up;
	for (int f = 0; f != NF; ++f) {
		const T slp_p = minmod(up2[f] - up1[f], up1[f] - u00[f]);
		const T slp_0 = minmod(up1[f] - u00[f], u00[f] - um1[f]);
		const T slp_m = minmod(u00[f] - um1[f], um1[f] - um2[f]);
		um.L[f] = um1[f] + 0.5 * slp_m;
		um.R[f] = u00[f] - 0.5 * slp_0;
		up.L[f] = u00[f] + 0.5 * slp_0;
		up.R[f] = up1[f] - 0.5 * slp_p;
	}
	const auto fp = up.flux(dim, ap);
	const auto fm = um.flux(dim, am);
	a = max(ap, am);
	for (int f = 0; f != NF; ++f) {
		dU[f] -= (fp[f] - fm[f]) * c0;
	}
	return a / dx[dim];
}

#endif /* HYDRO_HPP_ */
