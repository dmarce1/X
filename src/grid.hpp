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

class grid {
private:
	real fgamma;
	const int size;
	const std::array<int, NDIM> dims;
	const std::array<real, NDIM> dx;
	std::vector<real> U;
	std::vector<state_var<real>> dU;
public:
	int index(int i, int j, int k) const;
	real x(int i) const;
	real y(int i) const;
	real z(int i) const;
	grid(int nx, int ny, int nz, double spanx, double spany, double spanz);
	void initialize(initial_value_type type);
	void output(std::string filename) const;
	real step();
	~grid();
};

#endif /* GRID_HPP_ */
