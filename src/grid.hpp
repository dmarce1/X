/*
 * grid.hpp
 *
 *  Created on: Nov 3, 2017
 *      Author: dmarce1
 */

#ifndef GRID_HPP_
#define GRID_HPP_

#include "defs.hpp"
#include "initial.hpp"
#include <array>

enum grid_exec_policy {
	CPU, GPU
};

class grid {
private:
	real fgamma;
	const int size;
	const std::array<int, NDIM> dims;
	const std::array<real, NDIM> dX;
	thrust::host_vector<real> U;
	thrust::host_vector<state_var<real>> dU;
	thrust::host_vector<real> a;
	thrust::device_vector<real> gpu_U;
	thrust::device_vector<state_var<real>> gpu_dU;
	thrust::device_vector<real> gpu_a;
	const int& nx;
	const int& ny;
	const int& nz;
	const real& dx;
	const real& dy;
	const real& dz;
	void hydro_boundary_call(grid_exec_policy);
	void hydro_kernel(grid_exec_policy, int rk);
	void hydro_compute_u(grid_exec_policy, real dt, int rk);
	real max_speed(grid_exec_policy);
public:
	int index(int i, int j, int k) const;
	real x(int i) const;
	real y(int i) const;
	real z(int i) const;
	grid(int nx, int ny, int nz, double spanx, double spany, double spanz);
	void initialize(initial_value_type type);
	void output(std::string filename) const;
	real step(grid_exec_policy policy = GPU);

	~grid();
};

#endif /* GRID_HPP_ */
