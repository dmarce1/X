/*
 * grid.hpp
 *
 *  Created on: Nov 3, 2017
 *      Author: dmarce1
 */

#ifndef SUB_GRID_HPP_
#define SUB_GRID_HPP_

#include "defs.hpp"
#include "initial.hpp"
#include "state.hpp"
#include <array>

enum sub_grid_exec_policy {
	CPU, GPU
};

class grid;

class sub_grid {
private:
	real fgamma;
	const int size;
	const std::array<int, NDIM> dims;
	const std::array<real, NDIM> dX;
	thrust::host_vector<real> U;
	thrust::host_vector<state_var<real>> dU;
	const int& nx;
	const int& ny;
	const int& nz;
	const real& dx;
	const real& dy;
	const real& dz;
	real hydro_kernel(sub_grid_exec_policy, int rk);
	void hydro_compute_u(sub_grid_exec_policy, real dt, int rk);
public:
	friend class grid;
	int index(int i, int j, int k) const;
	real x(int i) const;
	real y(int i) const;
	real z(int i) const;
	sub_grid(int nx, int ny, int nz, double spanx, double spany, double spanz);

	~sub_grid();
};

#endif /* SUB_GRID_HPP_ */
