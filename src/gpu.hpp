/*
 * gpu.hpp
 *
 *  Created on: Nov 3, 2017
 *      Author: dmarce1
 */

#ifndef GPU_HPP_
#define GPU_HPP_

#include "defs.hpp"
#include "state.hpp"

__global__
void hydro_gpu_kernel(real* U, state_var<real>* dU, real* aret, int nx, int ny,
		int nz, real dx, real dy, real dz, int rk);

__global__
void hydro_gpu_compute_u(real* U, state_var<real>* dU, int nx, int ny, int nz,
		real dt, int rk);

#endif /* GPU_HPP_ */
