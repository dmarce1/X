#include "gpu.hpp"
#include "hydro.hpp"

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


__global__
void hydro_gpu_compute_u(real* U, state_var<real>* dU, int nx, int ny, int nz, real dt, int rk) {
	int N[NDIM] = { nx, ny, nz };
	const int xi = threadIdx.x + BW;
	const int yi = blockIdx.x + BW;
	const int zi = blockIdx.y + BW;
	hydro_compute_u(U, dU, xi, yi, zi, N, dt, rk);
}
