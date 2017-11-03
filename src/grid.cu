#include <silo.h>
#include "grid.hpp"
#include "cpu.hpp"
#include "gpu.hpp"



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


int grid::index(int i, int j, int k) const {
	return i + dims[XDIM] * (j + dims[YDIM] * k);
}

real grid::x(int i) const {
	return dx[XDIM] * (real(i) + 0.5);
}

real grid::y(int i) const {
	return dx[YDIM] * (real(i) + 0.5);
}

real grid::z(int i) const {
	return dx[ZDIM] * (real(i) + 0.5);
}

grid::grid(int nx, int ny, int nz, double spanx, double spany, double spanz) :
		fgamma(5.0 / 3.0), size(nx * ny * nz), dims( { nx, ny, nz }), dx( { spanx / real(nx), spany / real(ny), spanz / real(nz) }), U(NF * size), dU(size) {
}

void grid::initialize(initial_value_type type) {
	fgamma = problem_gamma(type);
	for (int k = 0; k < dims[ZDIM]; ++k) {
		for (int j = 0; j < dims[YDIM]; ++j) {
			for (int i = 0; i < dims[XDIM]; ++i) {
				auto var = initial_value(type, x(i), y(j), z(k));
				set_state_var(U.data(), index(i, j, k), var, size);
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
				coords[0][iii] = xi * dx[XDIM];
				coords[1][iii] = yi * dx[YDIM];
				coords[2][iii] = zi * dx[ZDIM];
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

real grid::step() {
	static device_vector<real> a;
	a.resize(size);
	device_vector<real> d_U(U);
	device_vector<state_var<real>> d_dU(dU);

	hydro_kernel(U, dU, a, dims[XDIM], dims[YDIM], dims[ZDIM], dx[XDIM], dx[YDIM], dx[ZDIM], 0);
	const auto amax = max_ele(a.begin(), a.end());
	const auto dt = 0.4 / amax;
	hydro_compute_u(U, dU, dims[XDIM], dims[YDIM], dims[ZDIM], dt, 0);
	hydro_boundary_call(U, dims[XDIM], dims[YDIM], dims[ZDIM]);
	hydro_kernel(U, dU, a, dims[XDIM], dims[YDIM], dims[ZDIM], dx[XDIM], dx[YDIM], dx[ZDIM], 1);
	hydro_compute_u(U, dU, dims[XDIM], dims[YDIM], dims[ZDIM], dt, 1);
	hydro_boundary_call(U, dims[XDIM], dims[YDIM], dims[ZDIM]);
#ifdef USE_GPU
	cudaThreadSynchronize();
#endif
	return dt;
}

grid::~grid() {
}
