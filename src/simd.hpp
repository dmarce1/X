/*
 * simd.hpp
 *
 *  Created on: Nov 2, 2017
 *      Author: dmarce1
 */

#ifndef SIMD_HPP_
#define SIMD_HPP_

static constexpr real zero = 0.0;
static constexpr real half = 0.5;
static constexpr real one = 1.0;

#ifndef __CUDA_ARCH__
template<>
struct constants<simd> {
	static constexpr simd zero = {0, 0, 0, 0};
};

inline simd pow(const simd& a, const real& b) {
	simd c;
	for (int i = 0; i != NDIM; ++i) {
		c[i] = pow(a[i], b);
	}
	return c;
}

inline real max(const simd& a) {
	real m = a[0];
	for( int i = 1; i < simd_size; ++i) {
		m = max(m,a[i]);
	}
	return m;
}

inline simd sqrt(const simd& a) {
	return _mm256_sqrt_pd(a);
}

inline simd min(const simd& a, const simd& b) {
	return _mm256_min_pd(a, b);
}

inline simd max(const simd& a, const simd& b) {
	return _mm256_max_pd(a, b);
}

inline simd abs(const simd& a) {
	return max(a, -a);
}

static simd copysign(const real& x, const simd& y) {
	static constexpr simd zero = {0.0, 0.0, 0.0, 0.0};
	static constexpr simd one = {1.0,1.0,1.0,1.0};
	static constexpr simd negtwo = {-2.0,-2.0,-2.0,-2.0};
	const auto isgn = _mm256_cmp_pd(zero, y, _CMP_GE_OS);
	const auto sgn = _mm256_and_pd(isgn, negtwo) + one;
	return sgn * x;
}

inline simd minmod(const simd& a, const simd& b) {
	const simd sgn = copysign(half, a) + copysign(half, b);
	return sgn * min(abs(a), abs(b));
}

#endif

#endif /* SIMD_HPP_ */
