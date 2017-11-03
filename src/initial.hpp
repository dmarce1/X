/*
 * initial.hpp
 *
 *  Created on: Nov 3, 2017
 *      Author: dmarce1
 */

#ifndef INITIAL_HPP_
#define INITIAL_HPP_

#include "defs.hpp"
#include "state.hpp"

enum initial_value_type {
	SOD
};

state_var<real> initial_value(initial_value_type, real x, real y, real z);
real problem_gamma(initial_value_type);

#endif /* INITIAL_HPP_ */
