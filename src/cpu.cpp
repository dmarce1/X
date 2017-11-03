#define _CXX_SOURCE

#include "cpu.hpp"
#include "hydro.hpp"

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
