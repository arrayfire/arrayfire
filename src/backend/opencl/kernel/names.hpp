/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <ops.hpp>
template<af_op_t T> static const char *binOpName() { return "ADD_OP"; }

template<> STATIC_ const char *binOpName<af_add_t>() { return "ADD_OP"; }
template<> STATIC_ const char *binOpName<af_mul_t>() { return "MUL_OP"; }
template<> STATIC_ const char *binOpName<af_and_t>() { return "AND_OP"; }
template<> STATIC_ const char *binOpName<af_or_t >() { return "OR_OP" ; }
template<> STATIC_ const char *binOpName<af_min_t>() { return "MIN_OP"; }
template<> STATIC_ const char *binOpName<af_max_t>() { return "MAX_OP"; }
template<> STATIC_ const char *binOpName<af_notzero_t>() { return "NOTZERO_OP"; }
