#include <silo.h>
#include "grid.hpp"
#include "cpu.hpp"
#include "gpu.hpp"

void grid::hydro_boundary_call(grid_exec_policy policy) {
	switch (policy) {
	case CPU:
		hydro_cpu_boundaries(U.data(), nx, ny, nz);
		break;
	case GPU:
		hydro_gpu_x_boundaries<<<nz, ny>>>(gpu_U.data().get(),nx,ny,nz);
		hydro_gpu_y_boundaries<<<nz, nx>>>(gpu_U.data().get(),nx,ny,nz);
		hydro_gpu_z_boundaries<<<ny, nx>>>(gpu_U.data().get(),nx,ny,nz);
		cudaThreadSynchronize();
		break;
	}
}

void grid::hydro_kernel(grid_exec_policy policy, int rk) {
	switch (policy) {
	case CPU:

		for (int dim = 0; dim < NDIM; ++dim) {
			static std::vector<real> u;
			static std::vector<state_var<real>> du;
			static std::vector<float> this_a(size);

			u.resize(NF * size);
			du.resize(size);
			this_a.resize(size);

			for (int f = 0; f != NF; ++f) {
				for (int i = 0; i != size; ++i) {
					u[i + f * dims[dim]] = U[map[dim][i] + f * size];
				}
			}
			for (int i = 0; i != size; ++i) {
				du[i] = dU[map[dim][i]];
			}

			hydro_cpu_kernel(u.data(), du.data(), this_a.data(), dims[dim], dim, dX[dim], rk);

			for (int i = 0; i != size; ++i) {
				dU[map[dim][i]] = du[i];
				a[map[dim][i]] = this_a[i];
			}

		}
		break;
	case GPU:
		//dim3 threads(WARP_SIZE);
//		dim3 blocks((nx - 2 * BW) / WARP_SIZE, ny - 2 * BW, nz - 2 * BW);
//		hydro_gpu_kernel<<<blocks,threads>>>(gpu_U.data().get(), gpu_dU.data().get(), gpu_a.data().get(), nx, ny, nz, dx, dy, dz, rk );
//		cudaThreadSynchronize();
		break;
	}
}

void grid::hydro_compute_u(grid_exec_policy policy, real dt, int rk) {
	switch (policy) {
	case CPU:
		hydro_cpu_compute_u(U.data(), dU.data(), nx, ny, nz, dt, rk);
		break;
	case GPU:
		dim3 threads(nx - 2 * BW);
		dim3 blocks(ny - 2 * BW, nz - 2 * BW);
		hydro_gpu_compute_u<<<blocks,threads>>>(gpu_U.data().get(), gpu_dU.data().get(), nx, ny, nz, dt, rk);
		cudaThreadSynchronize();
		break;
	}
}

int grid::index(int i, int j, int k) const {
	return i + dims[XDIM] * (j + dims[YDIM] * k);
}

real grid::x(int i) const {
	return dx * (real(i) + 0.5);
}

real grid::y(int i) const {
	return dy * (real(i) + 0.5);
}

real grid::z(int i) const {
	return dz * (real(i) + 0.5);
}

grid::grid(int _nx, int _ny, int _nz, double spanx, double spany, double spanz) :
		fgamma(5.0 / 3.0), size(_nx * _ny * _nz), dims( { _nx, _ny, _nz }), dX( { spanx / real(_nx), spany / real(_ny), spanz / real(_nz) }), U(NF * size), dU(
				size), a(size, 0.0), gpu_U(NF * size), gpu_dU(size), gpu_a(size), map(), nx(dims[XDIM]), ny(dims[YDIM]), nz(dims[ZDIM]), dx(dX[XDIM]), dy(
				dX[YDIM]), dz(dX[ZDIM]) {
	for (int dim = 0; dim != NDIM; ++dim) {
		map[dim].resize(size);
	}
	std::array<int, NDIM> n;
	for (n[ZDIM] = 0; n[ZDIM] < nz; n[ZDIM]++) {
		for (n[YDIM] = 0; n[YDIM] < ny; n[YDIM]++) {
			for (n[XDIM] = 0; n[XDIM] < nx; n[XDIM]++) {
				const int ix = n[XDIM] + nx * (n[YDIM] + ny * n[ZDIM]);
				const int iy = n[YDIM] + ny * (n[XDIM] + nx * n[ZDIM]);
				const int iz = n[ZDIM] + nz * (n[YDIM] + ny * n[XDIM]);
				const int i0 = ix;
				map[XDIM][i0] = ix;
				map[YDIM][i0] = iy;
				map[ZDIM][i0] = iz;
			}
		}
	}
}

void grid::initialize(initial_value_type type) {
	fgamma = problem_gamma(type);
	for (int k = 0; k < dims[ZDIM]; ++k) {
		for (int j = 0; j < dims[YDIM]; ++j) {
			for (int i = 0; i < dims[XDIM]; ++i) {
				auto var = initial_value(type, x(i), y(j), z(k));
				set_state_var(U.data(), index(i, j, k), var, size);
				dU[index(i, j, k)].zero();
			}
		}
	}
}

void grid::output(std::string filename) const {

	int sz_coord = (dims[XDIM] + 1) * (dims[YDIM] + 1) * (dims[ZDIM] + 1);
	real* coords[NDIM];
	char* coordnames[NDIM];
	int dims_coord[] = { dims[XDIM] + 1, dims[YDIM] + 1, dims[ZDIM] + 1 };
	for (int dim = 0; dim != NDIM; ++dim) {
		coords[dim] = new real[sz_coord];
		coordnames[dim] = new char[2];
		coordnames[dim][0] = 'x' + dim;
		coordnames[dim][1] = '\0';
	}
	for (int xi = 0; xi != dims[XDIM] + 1; ++xi) {
		for (int yi = 0; yi != dims[YDIM] + 1; ++yi) {
			for (int zi = 0; zi != dims[ZDIM] + 1; ++zi) {
				const int iii = xi + (dims[XDIM] + 1) * (yi + (dims[YDIM] + 1) * zi);
				coords[0][iii] = xi * dx;
				coords[1][iii] = yi * dy;
				coords[2][iii] = zi * dz;
			}
		}
	}
	auto db = DBCreate(filename.c_str(), DB_CLOBBER, DB_LOCAL, "CUDA Hydro", DB_PDB);

	DBPutQuadmesh(db, "mesh", coordnames, coords, dims_coord, NDIM, DB_DOUBLE, DB_NONCOLLINEAR, NULL);

	char** names = nullptr;
	names = new char*[NF];
	for (int f = 0; f != NF; ++f) {
		const auto n = state_var<real>::name(f);
		names[f] = new char[n.size() + 1];
		strcpy(names[f], n.c_str());
	}
	for (int f = 0; f != NF; ++f) {
		real* data_ptr = const_cast<real*>(U.data() + f * size);
		int* dims_ptr = const_cast<int*>(dims.data());
		DBPutQuadvar1(db, names[f], "mesh", data_ptr, dims_ptr, NDIM, NULL, 0, DB_DOUBLE, DB_ZONECENT, NULL);
	}
	DBClose(db);
	for (int dim = 0; dim != NDIM; ++dim) {
		delete[] coords[dim];
		delete[] coordnames[dim];
	}
	for (int f = 0; f != NF; ++f) {
		delete[] names[f];
	}
	delete[] names;
}

float grid::max_speed(grid_exec_policy policy) {
	real amax;
	switch (policy) {
	case GPU:
		amax = *thrust::max_element(gpu_a.begin(), gpu_a.end());
		break;
	case CPU:
		amax = *thrust::max_element(a.begin(), a.end());
		break;
	}
	return amax;
}

real grid::step(grid_exec_policy policy) {
	if (policy == GPU) {
		gpu_U = U;
	}
	hydro_kernel(policy, 0);
	const auto amax = max_speed(policy);
	const auto dt = real(0.4) / real(amax);
	hydro_compute_u(policy, dt, 0);
	hydro_boundary_call(policy);
	hydro_kernel(policy, 1);
	hydro_compute_u(policy, dt, 1);
	hydro_boundary_call(policy);
	if (policy == GPU) {
		U = gpu_U;
	}
	return dt;
}

grid::~grid() {
}
