#include <stdio.h>
#include <utility>
#include <vector>
#include <algorithm>
#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <numeric>
#include <silo.h>
#include <string>
#include "immintrin.h"
#include "state.hpp"


#include "hydro.hpp"


#define WARP_SIZE 32



template<>
struct constants<real> {
	static constexpr real zero = 0.0;
};

__global__
void hydro_gpu_x_boundaries(real* U, int nx, int ny, int nz) {
	const int i1 = threadIdx.x;
	const int i2 = blockIdx.x;
	hydro_x_boundaries(U, i1, i2, nx, ny, nz);
}

__global__
void hydro_gpu_y_boundaries(real* U, int nx, int ny, int nz) {
	const int i1 = threadIdx.x;
	const int i2 = blockIdx.x;
	hydro_y_boundaries(U, i1, i2, nx, ny, nz);
}

__global__
void hydro_gpu_z_boundaries(real* U, int nx, int ny, int nz) {
	const int i1 = threadIdx.x;
	const int i2 = blockIdx.x;
	hydro_z_boundaries(U, i1, i2, nx, ny, nz);
}

void hydro_cpu_boundaries(real* U, int nx, int ny, int nz) {
	for (int i1 = 0; i1 < nz; ++i1) {
#pragma omp parallel for
		for (int i2 = 0; i2 < ny; ++i2) {
			hydro_x_boundaries(U, i1, i2, nx, ny, nz);
		}
	}
	for (int i1 = 0; i1 < nz; ++i1) {
#pragma omp parallel for
		for (int i2 = 0; i2 < nx; ++i2) {
			hydro_y_boundaries(U, i1, i2, nx, ny, nz);
		}
	}

	for (int i1 = 0; i1 < ny; ++i1) {
#pragma omp parallel for
		for (int i2 = 0; i2 < nx; ++i2) {
			hydro_z_boundaries(U, i1, i2, nx, ny, nz);
		}
	}
}

inline EXPORT_GLOBAL
real minmod(const real& a, const real& b) {
	const real sgn = copysign(half, a) + copysign(half, b);
	return sgn * min(abs(a), abs(b));
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
	return a;
}

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
	/*if (rk == 1) {
		state_var<real> u00(U, idx, nx * ny * nz);
		T max_egas = u00.egas();
		const int idxxp = (xi + 1) + nx * (yi + ny * zi);
		const int idxxm = (xi - 1) + nx * (yi + ny * zi);
		const int idxyp = xi + nx * ((yi + 1) + ny * zi);
		const int idxym = xi + nx * ((yi - 1) + ny * zi);
		const int idxzp = xi + nx * (yi + ny * (zi + 1));
		const int idxzm = xi + nx * (yi + ny * (zi - 1));
		max_egas = max(max_egas, U[egas_i * sz + idxxp]);
		max_egas = max(max_egas, U[egas_i * sz + idxxm]);
		max_egas = max(max_egas, U[egas_i * sz + idxyp]);
		max_egas = max(max_egas, U[egas_i * sz + idxym]);
		max_egas = max(max_egas, U[egas_i * sz + idxzp]);
		max_egas = max(max_egas, U[egas_i * sz + idxzm]);
		const T eint = u00.egas() - u00.ekinetic();
		T sw = T(eint > T(0.1) * max_egas);
		u00.tau() += sw * (pow(eint, 1.0 / FGAMMA) - u00.tau());
		U[tau_i * sz + idx] = u00.tau();
	}*/
}

__global__
void hydro_gpu_kernel(real* U, state_var<real>* dU, real* aret, int nx, int ny, int nz, real dx, real dy, real dz, int rk) {
	real dX[NDIM] = { dx, dy, dz };
	int D[NDIM] = { 1, nx, nx * ny };
	const int xi = blockIdx.x * WARP_SIZE + threadIdx.x + BW;
	const int yi = blockIdx.y + BW;
	const int zi = blockIdx.z + BW;
	const int idx = xi + nx * (yi + ny * zi);
	real a = 0.0;
	const int sz = nx * ny * nz;
	for (int dim = 0; dim != NDIM; ++dim) {
		state_var<real> up1(U, idx + D[dim], sz);
		state_var<real> up2(U, idx + 2 * D[dim], sz);
		state_var<real> um1(U, idx - D[dim], sz);
		state_var<real> um2(U, idx - 2 * D[dim], sz);
		const real this_a = hydro_compute_du(state_var<real>(U, idx, sz), um2, um1, up1, up2, dU[idx], dX, rk, dim);
		a = max(this_a, a);
	}
	if (aret) {
		aret[idx] = a;
	}
}

#ifndef __CUDA_ARCH__

void hydro_cpu_kernel(real* U, state_var<real>* dU, real* aret, int nx, int ny, int nz, real dx, real dy, real dz, int rk) {
	int D[NDIM] = {1, nx, nx * ny};
	real dX[NDIM] = {dx, dy, dz};
	const int sz = nx * ny * nz;
	for (int zi = BW; zi < nz - BW; ++zi) {
		for (int yi = BW; yi < ny - BW; ++yi) {
#pragma omp parallel for
			for (int xi = BW; xi < nx - BW; xi += simd_size) {
				const int idx = xi + nx * (yi + ny * zi);
				state_var<simd> u0;
				state_var<simd> du;
				state_var<simd> up1;
				state_var<simd> up2;
				state_var<simd> um1;
				state_var<simd> um2;
				for (int f = 0; f != NF; ++f) {
					for (int i = 0; i < simd_size; ++i) {
						u0[f][i] = U[idx + i+sz*f];
						du[f][i] = dU[idx + i][f];
					}
				}
				simd a = constants<simd>::zero;
				for (int dim = 0; dim != NDIM; ++dim) {
					for (int f = 0; f != NF; ++f) {
						for (int i = 0; i < simd_size; ++i) {
							up1[f][i] = U[idx + D[dim] + i+sz*f];
							up2[f][i] = U[idx + 2 * D[dim] + i+sz*f];
							um1[f][i] = U[idx - D[dim] + i+sz*f];
							um2[f][i] = U[idx - 2 * D[dim] + i+sz*f];
						}
					}
					const auto this_a = hydro_compute_du<simd>(u0, um2, um1, up1, up2, du, dX, rk, dim);
					a = max(a, this_a);
				}
				for (int i = 0; i < simd_size; ++i) {
					for (int f = 0; f != NF; ++f) {
						dU[idx + i][f] = du[f][i];
					}
				}
				if (aret) {
					aret[idx/simd_size] = max(a);
				}
			}
		}
	}
}
#endif

__global__
void hydro_gpu_compute_u(real* U, state_var<real>* dU, int nx, int ny, int nz, real dt, int rk) {
	int N[NDIM] = { nx, ny, nz };
	const int xi = threadIdx.x + BW;
	const int yi = blockIdx.x + BW;
	const int zi = blockIdx.y + BW;
	hydro_compute_u(U, dU, xi, yi, zi, N, dt, rk);
}

void hydro_cpu_compute_u(real* U, state_var<real>* dU, int nx, int ny, int nz, real dt, int rk) {
	int N[NDIM] = { nx, ny, nz };
	for (int zi = BW; zi < nz - BW; ++zi) {
		for (int yi = BW; yi < ny - BW; ++yi) {
#pragma omp parallel for
			for (int xi = BW; xi < nx - BW; ++xi) {
				hydro_compute_u(U, dU, xi, yi, zi, N, dt, rk);
			}
		}
	}
}

void hydro_boundary_call(device_vector<real>& U, int nx, int ny, int nz) {
#ifdef USE_CPU
	hydro_cpu_boundaries(U.data(), nx, ny, nz);
#else
	hydro_gpu_x_boundaries<<<nz, ny>>>(U.data().get(),nx,ny,nz);
	hydro_gpu_y_boundaries<<<nz, nx>>>(U.data().get(),nx,ny,nz);
	hydro_gpu_z_boundaries<<<ny, nx>>>(U.data().get(),nx,ny,nz);
#endif
}

void hydro_kernel(device_vector<real>& U, device_vector<state_var<real>>& dU, device_vector<double>& a, int nx, int ny, int nz, real dx, real dy, real dz,
		int rk) {
#ifdef USE_CPU
	hydro_cpu_kernel(U.data(), dU.data(), a.data(), nx, ny, nz, dx, dy, dz, rk);
#else
	dim3 threads(WARP_SIZE);
	dim3 blocks((nx - 2 * BW) / WARP_SIZE, ny - 2 * BW, nz - 2 * BW);
	hydro_gpu_kernel<<<blocks,threads>>>(U.data().get(), dU.data().get(), a.data().get(), nx, ny, nz, dx, dy, dz, rk );
#endif
}

void hydro_compute_u(device_vector<real>& U, device_vector<state_var<real>>& dU, int nx, int ny, int nz, real dt, int rk) {
#ifdef USE_CPU
	hydro_cpu_compute_u(U.data(), dU.data(), nx, ny, nz, dt, rk);
#else
	dim3 threads(nx - 2 * BW);
	dim3 blocks(ny - 2 * BW, nz - 2 * BW);
	hydro_gpu_compute_u<<<blocks,threads>>>(U.data().get(), dU.data().get(), nx, ny, nz, dt, rk);
#endif
}

real hydro_step(device_vector<real>& U, device_vector<state_var<real>>& dU, int nx, int ny, int nz, real dx) {
	static device_vector<real> a(nx * ny * nz);
	hydro_kernel(U, dU, a, nx, ny, nz, dx, dx, dx, 0);
	const auto amax = max_ele(a.begin(), a.end());
	const auto dt = 0.4 * dx / amax;
	hydro_compute_u(U, dU, nx, ny, nz, dt, 0);
	hydro_boundary_call(U, nx, ny, nz);
	hydro_kernel(U, dU, a, nx, ny, nz, dx, dx, dx, 1);
	hydro_compute_u(U, dU, nx, ny, nz, dt, 1);
	hydro_boundary_call(U, nx, ny, nz);
#ifdef USE_GPU
	cudaThreadSynchronize();
#endif
	return reinterpret_cast<const real*>(&dt)[0];
}

device_vector<real> hydro_init(int nx, int ny, int nz) {
	host_vector<real> U(NF * nx * ny * nz);
	const int D = nx * ny * nz;
	for (int zi = 0; zi < nz; ++zi) {
		for (int yi = 0; yi < ny; ++yi) {
			for (int xi = 0; xi < nx; ++xi) {
				const int idx = xi + nx * (yi + ny * zi);
				for (int f = 0; f != NF; ++f) {
					U[f * D + idx] = 0.0;
				}
				if (xi < nx / 2) {
					U[rho_i * D + idx] = 1.0;
					U[egas_i * D + idx] = 1.0;
				} else {
					U[rho_i * D + idx] = 0.125;
					U[egas_i * D + idx] = 0.25;
				}
			}
		}
	}
	return U;
}

void write_silo(const device_vector<real>& U0, const char* filename, int nx, int ny, int nz, real dx, real dy, real dz) {
	const host_vector<real> U(U0);
	int sz = nx * ny * nz;
	int sz_coord = (nx + 1) * (ny + 1) * (nz + 1);
	real* coords[NDIM];
	char* coordnames[NDIM];
	int dims[] = { nx, ny, nz };
	int dims_coord[] = { nx + 1, ny + 1, nz + 1 };
	for (int dim = 0; dim != NDIM; ++dim) {
		coords[dim] = new real[sz_coord];
		coordnames[dim] = new char[2];
		coordnames[dim][0] = 'x' + dim;
		coordnames[dim][1] = '\0';
	}
	for (int xi = 0; xi != nx + 1; ++xi) {
		for (int yi = 0; yi != ny + 1; ++yi) {
			for (int zi = 0; zi != nz + 1; ++zi) {
				const int iii = xi + (nx + 1) * (yi + (ny + 1) * zi);
				coords[0][iii] = xi * dx;
				coords[1][iii] = yi * dy;
				coords[2][iii] = zi * dz;
			}
		}
	}
	auto db = DBCreate(filename, DB_CLOBBER, DB_LOCAL, "CUDA Hydro", DB_PDB);

	DBPutQuadmesh(db, "mesh", coordnames, coords, dims_coord, NDIM, DB_DOUBLE, DB_NONCOLLINEAR, NULL);

	static char** names = nullptr;
	if (names == nullptr) {
		names = new char*[NF];
		for (int f = 0; f != NF; ++f) {
			const auto n = state_var<real>::name(f);
			names[f] = new char[n.size() + 1];
			strcpy(names[f], n.c_str());
		}
	}

	real* tmp = new real[sz];
	for (int f = 0; f != NF; ++f) {
		for (int i = 0; i < nx * ny * nz; ++i) {
			tmp[i] = U[f * nx * ny * nz + i];
		}
		DBPutQuadvar1(db, names[f], "mesh", tmp, dims, NDIM, NULL, 0, DB_DOUBLE, DB_ZONECENT, NULL);
	}
	delete[] tmp;
	DBClose(db);
	for (int dim = 0; dim != NDIM; ++dim) {
		delete[] coords[dim];
		delete[] coordnames[dim];
	}
}

int main(void) {
	int nx = 256;
	int ny = 256;
	int nz = 256;
	real dx = 1.0 / nx;
	device_vector<real> U = hydro_init(nx, ny, nz);
	device_vector<state_var<real>> dU(nx * ny * nz);
	real t = 0.0;
	int i;
	for (i = 0; i < 10; ++i) {
		std::string fname = std::string("X.") + std::to_string(i) + std::string(".silo");
		//	write_silo(U, fname.c_str(), nx, ny, nz, dx, dx, dx);
		auto dt = hydro_step(U, dU, nx, ny, nz, dx);
		printf("%i %e %e\n", i, t, dt);
		t += dt;
	}
	std::string fname = std::string("X.") + std::to_string(i) + std::string(".silo");
//	write_silo(U, fname.c_str(), nx, ny, nz, dx, dx, dx);

	return 0;
}

