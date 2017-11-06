/*
 * grid.hpp
 *
 *  Created on: Nov 6, 2017
 *      Author: dmarce1
 */

#ifndef GRID_HPP_
#define GRID_HPP_

#include "defs.hpp"
#include "sub_grid.hpp"
#include <vector>


class grid {
private:
	real fgamma;
	const int nx;
	const int ny;
	const int nz;
	const int ndiv;
	double dx;
	double dy;
	double dz;
	const int size;
	const int dsize;
	std::vector<state_var<real>> U;
	std::vector<state_var<real>> dU;
	std::vector<std::array<std::pair<int, int>, NDIM>> divs;
	std::vector<std::shared_ptr<sub_grid>> subgrids;
	void copy_to(int) const;
	void copy_from(int);
	std::array<int,NDIM> dims;
	int div_index(int i, int j, int k) const {
		return i + ndiv * (j + ndiv * k);
	}
	int index(int i, int j, int k) const {
		return i + nx * (j + ny * k);
	}
public:
	real x(int) const;
	real y(int) const;
	real z(int) const;
	grid(int _nx, int _ny, int _nz, int div, double _dx, double _dy, double _dz);
	void initialize(initial_value_type type);
	void output(std::string filename) const;
	real step();
	void boundaries();

};

#endif /* GRID_HPP_ */
