/*
 * cpu.hpp
 *
 *  Created on: Nov 3, 2017
 *      Author: dmarce1
 */

#ifndef CPU_HPP_
#define CPU_HPP_


#include "defs.hpp"
#include "state.hpp"



void hydro_cpu_kernel(real* U, state_var<real>* dU, real* aret, int nx, int ny, int nz, real dx, real dy, real dz, int rk);
void hydro_cpu_boundaries(real* U, int nx, int ny, int nz);
void hydro_cpu_compute_u(real* U, state_var<real>* dU, int nx, int ny, int nz, real dt, int rk);


#endif /* CPU_HPP_ */
