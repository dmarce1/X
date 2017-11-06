#include "grid.hpp"
#include <silo.h>
#include <future>

grid::grid(int _nx, int _ny, int _nz, int div, double _dx, double _dy, double _dz) :
		fgamma(5.0 / 3.0), nx(_nx), ny(_ny), nz(_nz), ndiv(div), dx(_dx), dy(_dy), dz(_dz), size(nx * ny * nz), dsize(ndiv * ndiv * ndiv), U(size), dU(size), divs(
				dsize), subgrids(dsize), dims() {
	dims[XDIM] = nx;
	dims[YDIM] = ny;
	dims[ZDIM] = nz;
	for (int i = 0; i < ndiv; ++i) {
		for (int j = 0; j < ndiv; ++j) {
			for (int k = 0; k < ndiv; ++k) {
				const int idx = div_index(i, j, k);
				divs[idx][XDIM].first = (i * (nx - 2 * BW)) / ndiv;
				divs[idx][YDIM].first = (j * (nx - 2 * BW)) / ndiv;
				divs[idx][ZDIM].first = (k * (nx - 2 * BW)) / ndiv;
				divs[idx][XDIM].second = ((i + 1) * (nx - 2 * BW)) / ndiv + 2 * BW;
				divs[idx][YDIM].second = ((j + 1) * (nx - 2 * BW)) / ndiv + 2 * BW;
				divs[idx][ZDIM].second = ((k + 1) * (nx - 2 * BW)) / ndiv + 2 * BW;
			}
		}
	}
	for (int i = 0; i < dsize; ++i) {
		const int this_nx = divs[i][XDIM].second - divs[i][XDIM].first;
		const int this_ny = divs[i][YDIM].second - divs[i][YDIM].first;
		const int this_nz = divs[i][ZDIM].second - divs[i][ZDIM].first;
		subgrids[i] = std::make_shared<sub_grid>(this_nx, this_ny, this_nz, dx, dy, dz);
	}
}

void grid::copy_to(int i0) const {
	auto& sg = *subgrids[i0];
	for (int k = 0; k < sg.nz; ++k) {
		const int this_k = k + divs[i0][ZDIM].first;
		for (int j = 0; j < sg.ny; ++j) {
			const int this_j = j + divs[i0][YDIM].first;
			for (int i = 0; i < sg.nx; ++i) {
				const int this_i = i + divs[i0][XDIM].first;
				for (int f = 0; f != NF; ++f) {
					sg.U[sg.index(i, j, k) + f * sg.size] = U[index(this_i, this_j, this_k)][f];
				}
				sg.dU[sg.index(i, j, k)] = dU[index(this_i, this_j, this_k)];
			}
		}
	}
}

void grid::copy_from(int i0) {
	const auto& sg = *subgrids[i0];
	for (int k = BW; k < sg.nz - BW; ++k) {
		const int this_k = k + divs[i0][ZDIM].first;
		for (int j = BW; j < sg.ny - BW; ++j) {
			const int this_j = j + divs[i0][YDIM].first;
			for (int i = BW; i < sg.nx - BW; ++i) {
				const int this_i = i + divs[i0][XDIM].first;
				for (int f = 0; f != NF; ++f) {
					U[index(this_i, this_j, this_k)][f] = sg.U[sg.index(i, j, k) + f * sg.size];
				}
				dU[index(this_i, this_j, this_k)] = sg.dU[sg.index(i, j, k)];
			}
		}
	}
}

real grid::step() {
	const auto lp = std::launch::async;
	const int ncpu = 8;
	const int ngpu = 1;
	const int ntot = ncpu + ngpu;
	std::vector<std::future<real>> rfuts(ntot);
	std::vector<std::future<void>> vfuts(ntot);
	auto counter = std::make_shared<std::atomic<int>>(0);
	auto gpu_counter = std::make_shared<std::atomic<int>>(0);
	auto cpu_counter = std::make_shared<std::atomic<int>>(0);
	for (int i = 0; i != ntot; ++i) {
		rfuts[i] = std::async(lp, [=]() {
			const sub_grid_exec_policy policy = i < ncpu ? CPU : GPU;
			int d = (*counter)++;
			real amax = 0.0;
			while(d < dsize ) {
				auto subgrid = subgrids[d];
				copy_to(d);
				auto this_amax = subgrid->hydro_kernel(policy, 0);
				amax = std::max(amax,this_amax);
				d = (*counter)++;
				(*(policy == GPU ? gpu_counter : cpu_counter))++;
			}
			return amax;
		});
	}
	real amax = 0.0;
	for (auto& f : rfuts) {
		amax = std::max(f.get(), amax);
	}

	const auto dt = 0.4 / amax;
	*counter = 0;
	for (int i = 0; i != ntot; ++i) {
		vfuts[i] = std::async(lp, [=]() {
			int d = (*counter)++;
			const sub_grid_exec_policy policy = i < ncpu ? CPU : GPU;
			while(d < dsize ) {
				auto subgrid = subgrids[d];
				subgrid->hydro_compute_u(policy, dt, 0);
				d = (*counter)++;
				(*(policy == GPU ? gpu_counter : cpu_counter))++;
			}
		});
	}
	for (auto& f : vfuts) {
		f.get();
	}

	for (int d = 0; d != dsize; ++d) {
		copy_from(d);
	}
	boundaries();
	*counter = 0;
	for (int i = 0; i != ntot; ++i) {
		vfuts[i] = std::async(lp, [=]() {
			int d = (*counter)++;
			while(d < dsize ) {
				const sub_grid_exec_policy policy = i < ncpu ? CPU : GPU;
				auto subgrid = subgrids[d];
				copy_to(d);
				subgrid->hydro_kernel(policy, 1);
				subgrid->hydro_compute_u(policy, dt, 1);
				d = (*counter)++;
				(*(policy == GPU ? gpu_counter : cpu_counter))++;
			}
		});
	}
	for (auto& f : vfuts) {
		f.get();
	}

	for (int d = 0; d != dsize; ++d) {
		copy_from(d);
	}
	boundaries();
	printf( "%i %i\n", int(*gpu_counter), int(*cpu_counter));
	return dt;
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

void grid::initialize(initial_value_type type) {
	fgamma = problem_gamma(type);
	for (int k = 0; k < dims[ZDIM]; ++k) {
		for (int j = 0; j < dims[YDIM]; ++j) {
			for (int i = 0; i < dims[XDIM]; ++i) {
				U[index(i, j, k)] = initial_value(type, x(i), y(j), z(k));
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
	std::vector<double> tmp(size);
	for (int f = 0; f != NF; ++f) {
		for (int i = 0; i != size; ++i) {
			tmp[i] = U[i][f];
		}
		int* dims_ptr = const_cast<int*>(dims.data());
		DBPutQuadvar1(db, names[f], "mesh", tmp.data(), dims_ptr, NDIM, NULL, 0, DB_DOUBLE, DB_ZONECENT, NULL);
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

void grid::boundaries() {

	int i, j, k, i0, j0;

	for (int k = BW; k < nz; ++k) {
		for (int j = BW; j < ny; ++j) {
			i0 = BW + nx * (j + ny * k);
			j0 = (nx - BW - 1) + nx * (j + ny * k);
			for (int i = 1; i <= BW; ++i) {
				U[i0 - i] = U[i0];
				U[j0 + i] = U[j0];
				U[i0 - i].mom(0) = std::min(U[i0].mom(0), zero);
				U[j0 + i].mom(0) = std::max(U[j0].mom(0), zero);
			}
		}
	}
	for (int k = BW; k < nz; ++k) {
		for (int i = BW; i < nx; ++i) {
			i0 = i + nx * (BW + ny * k);
			j0 = i + nx * ((ny - BW - 1) + ny * k);
			for (int j = 1; j <= BW; ++j) {
				U[i0 - j * nx] = U[i0];
				U[j0 + j * nx] = U[j0];
				U[i0 - j * ny].mom(1) = std::min(U[i0].mom(1), zero);
				U[j0 + j * ny].mom(1) = std::max(U[j0].mom(1), zero);
			}
		}
	}
	for (int j = BW; j < ny; ++j) {
		for (int i = BW; i < nx; ++i) {
			i0 = i + nx * (j + ny * BW);
			j0 = i + nx * (j + ny * (nz - BW - 1));
			for (int k = 1; k <= BW; ++k) {
				U[i0 - k * nx * ny] = U[i0];
				U[j0 + k * nx * ny] = U[j0];
				U[i0 - k * nx * ny].mom(2) = std::min(U[i0].mom(2), zero);
				U[j0 + k * nx * ny].mom(2) = std::max(U[j0].mom(2), zero);
			}
		}
	}

}
