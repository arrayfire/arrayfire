/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#ifdef __cplusplus
#include <complex>
namespace af{
typedef std::complex<float> af_cfloat;
typedef std::complex<double> af_cdouble;
}
#else
typedef struct {
    float x;
    float y;
} af_cfloat;

typedef struct {
    double x;
    double y;
} af_cdouble;
#endif

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
    s8,
    u8x4,
    s8x4,
} af_dtype;

typedef enum {
    afDevice,
    afHost,
} af_source;

#if defined(_WIN32) || defined(_MSC_VER)
  // http://msdn.microsoft.com/en-us/library/b0084kay(v=VS.80).aspx
  // http://msdn.microsoft.com/en-us/library/3y1sfaz2%28v=VS.80%29.aspx
  #ifdef AFDLL // libaf
    #define AFAPI  __declspec(dllexport) extern "C"
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
#else
  #define AFAPI   __attribute__((visibility("default")))
  #include <stdbool.h>
#endif

#include <cstddef>
#define AF_MAX_DIMS 4

typedef size_t af_array;

typedef long long dim_type;

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
    AF_CONNECTIVITY_4 = 0,
    AF_CONNECTIVITY_8
} af_connectivity_type;
