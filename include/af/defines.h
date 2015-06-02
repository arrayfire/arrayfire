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
    #define SIZE_T_FRMT_SPECIFIER "%Iu"
    #define DEPRECATED(msg) __declspec(deprecated( msg ))
#else
    #define AFAPI   __attribute__((visibility("default")))
    #include <stdbool.h>
    #define __PRETTY_FUNCTION__ __func__
    #define STATIC_ inline
    #define SIZE_T_FRMT_SPECIFIER "%zu"
#if __GNUC__ >= 4 && __GNUC_MINOR > 4
    #define DEPRECATED(msg) __attribute__((deprecated( msg )))
#else
    #define DEPRECATED(msg) __attribute__((deprecated))
#endif

#endif

// Known 64-bit x86 and ARM architectures use long long
#if defined(__x86_64__) || defined(_M_X64) || defined(_WIN64) || defined(__aarch64__) || defined(__LP64__)   // 64-bit Architectures
    typedef long long   dim_t;
// Known 32-bit x86 and ARM architectures use int
#elif defined(__i386__) || defined(_M_IX86) || defined(__arm__) || defined(_M_ARM)     // 32-bit x86 Architecture
    typedef int         dim_t;
// All other platforms use long long
#else
    typedef long long   dim_t;
#endif

#include <stdlib.h>

typedef long long intl;
typedef unsigned long long uintl;

typedef enum {
    ///
    /// The function returned successfully
    ///
    AF_SUCCESS            =   0,

    // 100-199 Errors in environment

    ///
    /// The system or device ran out of memory
    ///
    AF_ERR_NO_MEM         = 101,

    ///
    /// There was an error in the device driver
    ///
    AF_ERR_DRIVER         = 102,

    ///
    /// There was an error with the runtime environment
    ///
    AF_ERR_RUNTIME        = 103,

    // 200-299 Errors in input parameters

    ///
    /// The input array is not a valid af_array object
    ///
    AF_ERR_INVALID_ARRAY  = 201,

    ///
    /// One of the function arguments is incorrect
    ///
    AF_ERR_ARG            = 202,

    ///
    /// The size is incorrect
    ///
    AF_ERR_SIZE           = 203,

    ///
    /// The type is not suppported by this function
    ///
    AF_ERR_TYPE           = 204,

    ///
    /// The type of the input arrays are not compatible
    ///
    AF_ERR_DIFF_TYPE      = 205,

    ///
    /// Function does not support GFOR / batch mode
    ///
    AF_ERR_BATCH          = 207,


    // 300-399 Errors for missing software features

    ///
    /// The option is not supported
    ///
    AF_ERR_NOT_SUPPORTED  = 301,

    ///
    /// This build of ArrayFire does not support this feature
    ///
    AF_ERR_NOT_CONFIGURED = 302,
    // 400-499 Errors for missing hardware features

    ///
    /// This device does not support double
    ///
    AF_ERR_NO_DBL         = 401,

    ///
    /// This build of ArrayFire was not built with graphics or this device does
    /// not support graphics
    ///
    AF_ERR_NO_GFX         = 402,
    // 900-999 Errors from upstream libraries and runtimes

    ///
    /// There was an internal error either in ArrayFire or in a project
    /// upstream
    ///
    AF_ERR_INTERNAL       = 998,

    ///
    /// Unknown Error
    ///
    AF_ERR_UNKNOWN        = 999
} af_err;

typedef enum {
    f32,    ///< 32-bit floating point values
    c32,    ///< 32-bit complex floating point values
    f64,    ///< 64-bit complex floating point values
    c64,    ///< 64-bit complex floating point values
    b8,     ///< 8-bit boolean values
    s32,    ///< 32-bit signed integral values
    u32,    ///< 32-bit unsigned integral values
    u8,     ///< 8-bit unsigned integral values
    s64,    ///< 64-bit signed integral values
    u64     ///< 64-bit unsigned integral values
} af_dtype;

typedef enum {
    afDevice,   ///< Device pointer
    afHost,     ///< Host pointer
} af_source;

#define AF_MAX_DIMS 4

// A handle for an internal array object
typedef void * af_array;

typedef enum {
    AF_INTERP_NEAREST,  ///< Nearest Interpolation
    AF_INTERP_LINEAR,   ///< Linear Interpolation
    AF_INTERP_BILINEAR, ///< Bilinear Interpolation
    AF_INTERP_CUBIC     ///< Cubic Interpolation
} af_interp_type;

typedef enum {
    ///
    /// Out of bound values are 0
    ///
    AF_PAD_ZERO = 0,

    ///
    /// Out of bound values are symmetric over the edge
    ///
    AF_PAD_SYM
} af_border_type;

typedef enum {
    ///
    /// Connectivity includes neighbors, North, East, South and West of current pixel
    ///
    AF_CONNECTIVITY_4 = 4,

    ///
    /// Connectivity includes 4-connectivity neigbors and also those on Northeast, Northwest, Southeast and Southwest
    ///
    AF_CONNECTIVITY_8 = 8
} af_connectivity;

typedef enum {

    ///
    /// Output of the convolution is the same size as input
    ///
    AF_CONV_DEFAULT,

    ///
    /// Output of the convolution is signal_len + filter_len - 1
    ///
    AF_CONV_EXPAND,
} af_conv_mode;

typedef enum {
    AF_CONV_AUTO,    ///< ArrayFire automatically picks the right convolution algorithm
    AF_CONV_SPATIAL, ///< Perform convolution in spatial domain
    AF_CONV_FREQ,    ///< Perform convolution in frequency domain
} af_conv_domain;

typedef enum {
    AF_SAD = 0,   ///< Match based on Sum of Absolute Differences (SAD)
    AF_ZSAD,      ///< Match based on Zero mean SAD
    AF_LSAD,      ///< Match based on Locally scaled SAD
    AF_SSD,       ///< Match based on Sum of Squared Differences (SSD)
    AF_ZSSD,      ///< Match based on Zero mean SSD
    AF_LSSD,      ///< Match based on Locally scaled SSD
    AF_NCC,       ///< Match based on Normalized Cross Correlation (NCC)
    AF_ZNCC,      ///< Match based on Zero mean NCC
    AF_SHD        ///< Match based on Sum of Hamming Distances (SHD)
} af_match_type;

typedef enum {
    AF_GRAY = 0, ///< Grayscale
    AF_RGB,      ///< 3-channel RGB
    AF_HSV       ///< 3-channel HSV
} af_cspace_t;

typedef enum {
    AF_MAT_NONE       = 0,    ///< Default
    AF_MAT_TRANS      = 1,    ///< Data needs to be transposed
    AF_MAT_CTRANS     = 2,    ///< Data needs to be conjugate tansposed
    AF_MAT_UPPER      = 32,   ///< Matrix is upper triangular
    AF_MAT_LOWER      = 64,   ///< Matrix is lower triangular
    AF_MAT_DIAG_UNIT  = 128,  ///< Matrix diagonal contains unitary values
    AF_MAT_SYM        = 512,  ///< Matrix is symmetric
    AF_MAT_POSDEF     = 1024, ///< Matrix is positive definite
    AF_MAT_ORTHOG     = 2048, ///< Matrix is orthogonal
    AF_MAT_TRI_DIAG   = 4096, ///< Matrix is tri diagonal
    AF_MAT_BLOCK_DIAG = 8192  ///< Matrix is block diagonal
} af_mat_prop;

typedef enum {
    AF_NORM_VECTOR_1,      ///< treats the input as a vector and returns the sum of absolute values
    AF_NORM_VECTOR_INF,    ///< treats the input as a vector and returns the max of absolute values
    AF_NORM_VECTOR_2,      ///< treats the input as a vector and returns euclidean norm
    AF_NORM_VECTOR_P,      ///< treats the input as a vector and returns the p-norm
    AF_NORM_MATRIX_1,      ///< return the max of column sums
    AF_NORM_MATRIX_INF,    ///< return the max of row sums
    AF_NORM_MATRIX_2,      ///< returns the max singular value). Currently NOT SUPPORTED
    AF_NORM_MATRIX_L_PQ,   ///< returns Lpq-norm

    AF_NORM_EUCLID = AF_NORM_VECTOR_2, ///< The default. Same as AF_NORM_VECTOR_2
} af_norm_type;

typedef enum {
    AF_COLORMAP_DEFAULT = 0,    ///< Default grayscale map
    AF_COLORMAP_SPECTRUM= 1,    ///< Spectrum map
    AF_COLORMAP_COLORS  = 2,    ///< Colors
    AF_COLORMAP_RED     = 3,    ///< Red hue map
    AF_COLORMAP_MOOD    = 4,    ///< Mood map
    AF_COLORMAP_HEAT    = 5,    ///< Heat map
    AF_COLORMAP_BLUE    = 6     ///< Blue hue map
} af_colormap;

// Below enum is purely added for example purposes
// it doesn't and shoudn't be used anywhere in the
// code. No Guarantee's provided if it is used.
typedef enum {
    AF_ID = 0
} af_someenum_t;

#ifdef __cplusplus
namespace af
{
    typedef af_dtype dtype;
    typedef af_source source;
    typedef af_interp_type interpType;
    typedef af_border_type borderType;
    typedef af_connectivity connectivity;
    typedef af_match_type matchType;
    typedef af_cspace_t CSpace;
    typedef af_someenum_t SomeEnum; // Purpose of Addition: How to add Function example
    typedef af_mat_prop trans;
    typedef af_conv_mode convMode;
    typedef af_conv_domain convDomain;
    typedef af_mat_prop matProp;
    typedef af_colormap ColorMap;
    typedef af_norm_type normType;
}

#endif
