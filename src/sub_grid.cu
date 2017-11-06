#include <silo.h>
#include "sub_grid.hpp"
#include "cpu.hpp"
#include "gpu.hpp"

real sub_grid::hydro_kernel(sub_grid_exec_policy policy, int rk) {
	real amax;
	switch (policy) {
	case CPU: {
		thrust::host_vector<real> a(size, 0.0);
		hydro_cpu_kernel(U.data(), dU.data(), a.data(), nx, ny, nz, dx, dy, dz, rk);
		if (rk == 0) {
			amax = *thrust::max_element(a.begin(), a.end());
		} else {
			amax = 0.0;
		}
		break;
	}
	case GPU: {
		dim3 threads(WARP_SIZE);
		dim3 blocks((nx - 2 * BW) / WARP_SIZE, ny - 2 * BW, nz - 2 * BW);
		thrust::device_vector<real> gpu_U = U;
		thrust::device_vector<state_var<real>> gpu_dU = dU;
		if (rk == 0) {
			thrust::device_vector<real> gpu_a(size, 0.0);
			hydro_gpu_kernel<<<blocks,threads>>>(gpu_U.data().get(), gpu_dU.data().get(), gpu_a.data().get(), nx, ny, nz, dx, dy, dz, rk );
			amax = *thrust::max_element(gpu_a.begin(), gpu_a.end());
		} else {
			hydro_gpu_kernel<<<blocks,threads>>>(gpu_U.data().get(), gpu_dU.data().get(), nullptr, nx, ny, nz, dx, dy, dz, rk );
			amax = 0.0;
		}
		dU = gpu_dU;
		break;
	}
	}
	return amax;
}

void sub_grid::hydro_compute_u(sub_grid_exec_policy policy, real dt, int rk) {
	switch (policy) {
	case CPU:
		hydro_cpu_compute_u(U.data(), dU.data(), nx, ny, nz, dt, rk);
		break;
	case GPU:
		dim3 threads(nx - 2 * BW);
		dim3 blocks(ny - 2 * BW, nz - 2 * BW);
		thrust::device_vector<real> gpu_U = U;
		thrust::device_vector<state_var<real>> gpu_dU = dU;
		hydro_gpu_compute_u<<<blocks,threads>>>(gpu_U.data().get(), gpu_dU.data().get(), nx, ny, nz, dt, rk);
//		cudaThreadSynchronize();
		U = gpu_U;
		dU = gpu_dU;
		break;
	}
}

int sub_grid::index(int i, int j, int k) const {
	return i + dims[XDIM] * (j + dims[YDIM] * k);
}

real sub_grid::x(int i) const {
	return dx * (real(i) + 0.5);
}

real sub_grid::y(int i) const {
	return dy * (real(i) + 0.5);
}

real sub_grid::z(int i) const {
	return dz * (real(i) + 0.5);
}

sub_grid::sub_grid(int _nx, int _ny, int _nz, double spanx, double spany, double spanz) :
		fgamma(5.0 / 3.0), size(_nx * _ny * _nz), dims( { _nx, _ny, _nz }), dX( { spanx, spany, spanz }), U(NF * size), dU(size), nx(dims[XDIM]), ny(
				dims[YDIM]), nz(dims[ZDIM]), dx(dX[XDIM]), dy(dX[YDIM]), dz(dX[ZDIM]) {
}

sub_grid::~sub_grid() {
}
