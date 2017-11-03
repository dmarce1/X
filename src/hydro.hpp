/*
 * hydro.hpp
 *
 *  Created on: Nov 3, 2017
 *      Author: dmarce1
 */

#ifndef HYDRO_HPP_
#define HYDRO_HPP_

#include "defs.hpp"

constexpr int BW = 2;

EXPORT_GLOBAL void hydro_x_boundaries(real* U, int i1, int i2, int nx, int ny, int nz);
EXPORT_GLOBAL void hydro_y_boundaries(real* U, int i1, int i2, int nx, int ny, int nz);
EXPORT_GLOBAL void hydro_z_boundaries(real* U, int i1, int i2, int nx, int ny, int nz);

#endif /* HYDRO_HPP_ */
