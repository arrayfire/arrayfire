/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#if defined(_WIN32) || defined(_MSC_VER)
	// http://msdn.microsoft.com/en-us/library/b0084kay(v=VS.80).aspx
	// http://msdn.microsoft.com/en-us/library/3y1sfaz2%28v=VS.80%29.aspx
	#ifdef AFDLL // libaf
		#define AFAPI  __declspec(dllexport)
	#else
		#define AFAPI  __declspec(dllimport)
	#endif

// bool
	#ifndef __cplusplus
		#define bool unsigned char
		#define false 0
		#define true  1
	#endif
	#define __PRETTY_FUNCTION__ __FUNCSIG__
	#define snprintf sprintf_s
	#define STATIC_ static
#else
	#define AFAPI   __attribute__((visibility("default")))
	#include <stdbool.h>
	#define __PRETTY_FUNCTION__ __func__
	#define STATIC_
#endif


#ifdef __cplusplus
#include <complex>
#include <cstddef>

typedef std::complex<float> af_cfloat;
typedef std::complex<double> af_cdouble;

#else
#include <stdlib.h>

typedef struct {
	float x;
	float y;
} af_cfloat;

typedef struct {
	double x;
	double y;
} af_cdouble;

#endif

typedef long long intl;
typedef unsigned long long uintl;

typedef enum {
    AF_SUCCESS=0,
    AF_ERR_INTERNAL,
    AF_ERR_NOMEM,
    AF_ERR_DRIVER,
    AF_ERR_RUNTIME,
    AF_ERR_INVALID_ARRAY,
    AF_ERR_ARG,
    AF_ERR_SIZE,
    AF_ERR_DIFF_TYPE,
    AF_ERR_NOT_SUPPORTED,
    AF_ERR_NOT_CONFIGURED,
    AF_ERR_INVALID_TYPE,
    AF_ERR_INVALID_ARG,
    AF_ERR_UNKNOWN
} af_err;

typedef enum {
	f32,
	c32,
	f64,
	c64,
	b8,
	s32,
	u32,
	u8,
    s64,
    u64
} af_dtype;

typedef enum {
	afDevice,
	afHost,
} af_source;

#define AF_MAX_DIMS 4

typedef size_t af_array;

typedef int dim_type;

typedef enum {
	AF_INTERP_NEAREST,
	AF_INTERP_LINEAR,
	AF_INTERP_BILINEAR,
	AF_INTERP_CUBIC
} af_interp_type;

typedef enum {
	AF_ZERO = 0,
	AF_SYMMETRIC
} af_pad_type;

typedef enum {
	AF_CONNECTIVITY_4 = 4,
	AF_CONNECTIVITY_8 = 8
} af_connectivity;

typedef enum {
    AF_SAD = 0,
    AF_ZSAD, // 1
    AF_LSAD, // 2
    AF_SSD,  // 3
    AF_ZSSD, // 4
    AF_LSSD, // 5
    AF_NCC,  // 6
    AF_ZNCC, // 7
    AF_SHD   // 8
} af_match_type;

#ifdef __cplusplus
#include <limits>
namespace af
{
    typedef af_cfloat cfloat;
    typedef af_cdouble  cdouble;
    typedef af_dtype dtype;
    typedef af_source source;
    typedef af_interp_type interpType;
    typedef af_pad_type padType;
    typedef af_connectivity connectivity;
    typedef af_match_type matchType;

    const double NaN = std::numeric_limits<double>::quiet_NaN();
    const double Inf = std::numeric_limits<double>::infinity();
    const double Pi = 3.1415926535897932384626433832795028841971693993751;
}

#endif
