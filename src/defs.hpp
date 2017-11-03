/*
 * defs.hpp
 *
 *  Created on: Nov 3, 2017
 *      Author: dmarce1
 */

#ifndef DEFS_HPP_
#define DEFS_HPP_



//#define USE_CPU
#define USE_GPU

#define EXPORT_GLOBAL __host__ __device__

using real = double;

static constexpr int NDIM = 3;

static constexpr real zero = real(0);
static constexpr real half = 0.5;
static constexpr real one = 1.0;



#endif /* DEFS_HPP_ */
