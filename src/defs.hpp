/*
 * defs.hpp
 *
 *  Created on: Nov 3, 2017
 *      Author: dmarce1
 */

#ifndef DEFS_HPP_
#define DEFS_HPP_

#include <vector>

#define WARP_SIZE 32

constexpr int XDIM = 0;
constexpr int YDIM = 1;
constexpr int ZDIM = 2;
constexpr int BW = 2;

#ifdef _CXX_SOURCE
#define EXPORT_GLOBAL
#else
#define EXPORT_GLOBAL __host__ __device__
#endif

using real = double;

static constexpr int NDIM = 3;

static constexpr real zero = real(0);
static constexpr real half = 0.5;
static constexpr real one = 1.0;

template<class T>
struct constants;

template<>
struct constants<real> {
	static constexpr real zero = 0.0;
};

#ifndef _CXX_SOURCE
#include <thrust/device_vector.h>
#endif

#endif /* DEFS_HPP_ */
