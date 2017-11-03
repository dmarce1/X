/*
 * defs.hpp
 *
 *  Created on: Nov 3, 2017
 *      Author: dmarce1
 */

#ifndef DEFS_HPP_
#define DEFS_HPP_

#define USE_CPU
//#define USE_GPU

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

#ifdef USE_CPU

template<class T>
using host_vector = std::vector<T>;

template<class T>
using device_vector = std::vector<T>;

template<class It>
real max_ele(It&& b, It&& e) {
	auto m = *b;
	for (It i = b + 1; i != e; ++i) {
		m = max(m, *i);
	}
	return m;
}

#else

#ifndef _CXX_SOURCE
template<class T>
using device_vector = thrust::device_vector<T>;

template<class T>
using host_vector = thrust::host_vector<T>;

template<class It>
real max_ele(It&& b, It&& e) {
	auto it = thrust::max_element(std::forward < It > (b),
			std::forward < It > (e));
	return *it;
}
#endif

#endif


#endif /* DEFS_HPP_ */
