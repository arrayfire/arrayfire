/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <complex>

namespace cpu
{
using cfloat  = std::complex<float>;
using cdouble = std::complex<double>;
using uint    = unsigned int;
using uchar   = unsigned char;
using ushort  = unsigned short;

#define lapack_complex_float  std::complex<float>
#define lapack_complex_double std::complex<double>

#ifdef USE_MKL
#define MKL_Complex8  std::complex<float>
#define MKL_Complex16 std::complex<double>
#endif

template<typename T> struct is_complex          { static const bool value = false;  };
template<> struct           is_complex<cfloat>  { static const bool value = true;   };
template<> struct           is_complex<cdouble> { static const bool value = true;   };
}
