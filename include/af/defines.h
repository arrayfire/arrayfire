/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#ifndef __CUDACC_RTC__
#include <af/compilers.h>
#endif

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
    #define SIZE_T_FRMT_SPECIFIER "%Iu"
    #define AF_DEPRECATED(msg) __declspec(deprecated( msg ))
    #if _MSC_VER >= 1800
        #define AF_HAS_VARIADIC_TEMPLATES
    #endif
#else
    #define AFAPI   __attribute__((visibility("default")))
    #include <stdbool.h>
    #define SIZE_T_FRMT_SPECIFIER "%zu"
#if __GNUC__ >= 4 && __GNUC_MINOR > 4
    #define AF_DEPRECATED(msg) __attribute__((deprecated( msg )))
#else
    #define AF_DEPRECATED(msg) __attribute__((deprecated))
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

#ifndef AFDLL  // prevents the use of these types internally
typedef AF_DEPRECATED("intl is deprecated. Use long long instead.") long long intl;
typedef AF_DEPRECATED("uintl is deprecated. Use unsigned long long instead.") unsigned long long uintl;
#endif

#include <af/version.h>
#ifndef AF_API_VERSION
#define AF_API_VERSION AF_API_VERSION_CURRENT
#endif

typedef enum {
    ///
    /// The function returned successfully
    ///
    AF_SUCCESS            =   0

    // 100-199 Errors in environment

    ///
    /// The system or device ran out of memory
    ///
    , AF_ERR_NO_MEM         = 101

    ///
    /// There was an error in the device driver
    ///
    , AF_ERR_DRIVER         = 102

    ///
    /// There was an error with the runtime environment
    ///
    , AF_ERR_RUNTIME        = 103

    // 200-299 Errors in input parameters

    ///
    /// The input array is not a valid af_array object
    ///
    , AF_ERR_INVALID_ARRAY  = 201

    ///
    /// One of the function arguments is incorrect
    ///
    , AF_ERR_ARG            = 202

    ///
    /// The size is incorrect
    ///
    , AF_ERR_SIZE           = 203

    ///
    /// The type is not suppported by this function
    ///
    , AF_ERR_TYPE           = 204

    ///
    /// The type of the input arrays are not compatible
    ///
    , AF_ERR_DIFF_TYPE      = 205

    ///
    /// Function does not support GFOR / batch mode
    ///
    , AF_ERR_BATCH          = 207


#if AF_API_VERSION >= 33
    ///
    /// Input does not belong to the current device.
    ///
    , AF_ERR_DEVICE         = 208
#endif

    // 300-399 Errors for missing software features

    ///
    /// The option is not supported
    ///
    , AF_ERR_NOT_SUPPORTED  = 301

    ///
    /// This build of ArrayFire does not support this feature
    ///
    , AF_ERR_NOT_CONFIGURED = 302

#if AF_API_VERSION >= 32
    ///
    /// This build of ArrayFire is not compiled with "nonfree" algorithms
    ///
    , AF_ERR_NONFREE        = 303
#endif

    // 400-499 Errors for missing hardware features

    ///
    /// This device does not support double
    ///
    , AF_ERR_NO_DBL         = 401

    ///
    /// This build of ArrayFire was not built with graphics or this device does
    /// not support graphics
    ///
    , AF_ERR_NO_GFX         = 402

#if AF_API_VERSION >= 37
    ///
    /// This device does not support half
    ///
    , AF_ERR_NO_HALF        = 403
#endif

    // 500-599 Errors specific to heterogenous API

#if AF_API_VERSION >= 32
    ///
    /// There was an error when loading the libraries
    ///
    , AF_ERR_LOAD_LIB       = 501
#endif

#if AF_API_VERSION >= 32
    ///
    /// There was an error when loading the symbols
    ///
    , AF_ERR_LOAD_SYM       = 502
#endif

#if AF_API_VERSION >= 32
    ///
    /// There was a mismatch between the input array and the active backend
    ///
    , AF_ERR_ARR_BKND_MISMATCH    = 503
#endif

    // 900-999 Errors from upstream libraries and runtimes

    ///
    /// There was an internal error either in ArrayFire or in a project
    /// upstream
    ///
    , AF_ERR_INTERNAL       = 998

    ///
    /// Unknown Error
    ///
    , AF_ERR_UNKNOWN        = 999
} af_err;

typedef enum {
    f32,    ///< 32-bit floating point values
    c32,    ///< 32-bit complex floating point values
    f64,    ///< 64-bit floating point values
    c64,    ///< 64-bit complex floating point values
    b8 ,    ///< 8-bit boolean values
    s32,    ///< 32-bit signed integral values
    u32,    ///< 32-bit unsigned integral values
    u8 ,    ///< 8-bit unsigned integral values
    s64,    ///< 64-bit signed integral values
    u64    ///< 64-bit unsigned integral values
#if AF_API_VERSION >= 32
    , s16    ///< 16-bit signed integral values
#endif
#if AF_API_VERSION >= 32
    , u16    ///< 16-bit unsigned integral values
#endif
#if AF_API_VERSION >= 37
    , f16    ///< 16-bit floating point value
#endif
} af_dtype;

typedef enum {
    afDevice,   ///< Device pointer
    afHost     ///< Host pointer
} af_source;

#define AF_MAX_DIMS 4

// A handle for an internal array object
typedef void * af_array;

typedef enum {
    AF_INTERP_NEAREST,         ///< Nearest Interpolation
    AF_INTERP_LINEAR,          ///< Linear Interpolation
    AF_INTERP_BILINEAR,        ///< Bilinear Interpolation
    AF_INTERP_CUBIC,           ///< Cubic Interpolation
    AF_INTERP_LOWER           ///< Floor Indexed
#if AF_API_VERSION >= 34
    , AF_INTERP_LINEAR_COSINE   ///< Linear Interpolation with cosine smoothing
#endif
#if AF_API_VERSION >= 34
    , AF_INTERP_BILINEAR_COSINE ///< Bilinear Interpolation with cosine smoothing
#endif
#if AF_API_VERSION >= 34
    , AF_INTERP_BICUBIC         ///< Bicubic Interpolation
#endif
#if AF_API_VERSION >= 34
    , AF_INTERP_CUBIC_SPLINE    ///< Cubic Interpolation with Catmull-Rom splines
#endif
#if AF_API_VERSION >= 34
    , AF_INTERP_BICUBIC_SPLINE  ///< Bicubic Interpolation with Catmull-Rom splines
#endif

} af_interp_type;

typedef enum {
    ///
    /// Out of bound values are 0
    ///
    AF_PAD_ZERO = 0,

    ///
    /// Out of bound values are symmetric over the edge
    ///
    AF_PAD_SYM,

    ///
    /// Out of bound values are clamped to the edge
    ///
    AF_PAD_CLAMP_TO_EDGE,

    ///
    /// Out of bound values are mapped to range of the dimension in cyclic fashion
    ///
    AF_PAD_PERIODIC
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
    AF_CONV_EXPAND
} af_conv_mode;

typedef enum {
    AF_CONV_AUTO,    ///< ArrayFire automatically picks the right convolution algorithm
    AF_CONV_SPATIAL, ///< Perform convolution in spatial domain
    AF_CONV_FREQ     ///< Perform convolution in frequency domain
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

#if AF_API_VERSION >= 31
typedef enum {
    AF_YCC_601 = 601,  ///< ITU-R BT.601 (formerly CCIR 601) standard
    AF_YCC_709 = 709,  ///< ITU-R BT.709 standard
    AF_YCC_2020 = 2020 ///< ITU-R BT.2020 standard
} af_ycc_std;
#endif

typedef enum {
    AF_GRAY = 0, ///< Grayscale
    AF_RGB,      ///< 3-channel RGB
    AF_HSV       ///< 3-channel HSV
#if AF_API_VERSION >= 31
    , AF_YCbCr     ///< 3-channel YCbCr
#endif
} af_cspace_t;

typedef enum {
    AF_MAT_NONE       = 0,    ///< Default
    AF_MAT_TRANS      = 1,    ///< Data needs to be transposed
    AF_MAT_CTRANS     = 2,    ///< Data needs to be conjugate tansposed
    AF_MAT_CONJ       = 4,    ///< Data needs to be conjugate
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

    AF_NORM_EUCLID = AF_NORM_VECTOR_2 ///< The default. Same as AF_NORM_VECTOR_2
} af_norm_type;

#if AF_API_VERSION >= 31
typedef enum {
    AF_FIF_BMP          = 0,    ///< FreeImage Enum for Bitmap File
    AF_FIF_ICO          = 1,    ///< FreeImage Enum for Windows Icon File
    AF_FIF_JPEG         = 2,    ///< FreeImage Enum for JPEG File
    AF_FIF_JNG          = 3,    ///< FreeImage Enum for JPEG Network Graphics File
    AF_FIF_PNG          = 13,   ///< FreeImage Enum for Portable Network Graphics File
    AF_FIF_PPM          = 14,   ///< FreeImage Enum for Portable Pixelmap (ASCII) File
    AF_FIF_PPMRAW       = 15,   ///< FreeImage Enum for Portable Pixelmap (Binary) File
    AF_FIF_TIFF         = 18,   ///< FreeImage Enum for Tagged Image File Format File
    AF_FIF_PSD          = 20,   ///< FreeImage Enum for Adobe Photoshop File
    AF_FIF_HDR          = 26,   ///< FreeImage Enum for High Dynamic Range File
    AF_FIF_EXR          = 29,   ///< FreeImage Enum for ILM OpenEXR File
    AF_FIF_JP2          = 31,   ///< FreeImage Enum for JPEG-2000 File
    AF_FIF_RAW          = 34    ///< FreeImage Enum for RAW Camera Image File
} af_image_format;
#endif

#if AF_API_VERSION >=34
typedef enum {
    AF_MOMENT_M00 = 1,
    AF_MOMENT_M01 = 2,
    AF_MOMENT_M10 = 4,
    AF_MOMENT_M11 = 8,
    AF_MOMENT_FIRST_ORDER = AF_MOMENT_M00 | AF_MOMENT_M01 | AF_MOMENT_M10 | AF_MOMENT_M11
} af_moment_type;
#endif

#if AF_API_VERSION >= 32
typedef enum {
    AF_HOMOGRAPHY_RANSAC = 0,   ///< Computes homography using RANSAC
    AF_HOMOGRAPHY_LMEDS  = 1    ///< Computes homography using Least Median of Squares
} af_homography_type;
#endif

#if AF_API_VERSION >= 32
// These enums should be 2^x
typedef enum {
    AF_BACKEND_DEFAULT = 0,  ///< Default backend order: OpenCL -> CUDA -> CPU
    AF_BACKEND_CPU     = 1,  ///< CPU a.k.a sequential algorithms
    AF_BACKEND_CUDA    = 2,  ///< CUDA Compute Backend
    AF_BACKEND_OPENCL  = 4   ///< OpenCL Compute Backend
} af_backend;
#endif

// Below enum is purely added for example purposes
// it doesn't and shoudn't be used anywhere in the
// code. No Guarantee's provided if it is used.
typedef enum {
    AF_ID = 0
} af_someenum_t;

#if AF_API_VERSION >=34
typedef enum {
    AF_BINARY_ADD  = 0,
    AF_BINARY_MUL  = 1,
    AF_BINARY_MIN  = 2,
    AF_BINARY_MAX  = 3
} af_binary_op;
#endif

#if AF_API_VERSION >=34
typedef enum {
    AF_RANDOM_ENGINE_PHILOX_4X32_10     = 100,                                  //Philox variant with N = 4, W = 32 and Rounds = 10
    AF_RANDOM_ENGINE_THREEFRY_2X32_16   = 200,                                  //Threefry variant with N = 2, W = 32 and Rounds = 16
    AF_RANDOM_ENGINE_MERSENNE_GP11213   = 300,                                  //Mersenne variant with MEXP = 11213
    AF_RANDOM_ENGINE_PHILOX             = AF_RANDOM_ENGINE_PHILOX_4X32_10,      //Resolves to Philox 4x32_10
    AF_RANDOM_ENGINE_THREEFRY           = AF_RANDOM_ENGINE_THREEFRY_2X32_16,    //Resolves to Threefry 2X32_16
    AF_RANDOM_ENGINE_MERSENNE           = AF_RANDOM_ENGINE_MERSENNE_GP11213,    //Resolves to Mersenne GP 11213
    AF_RANDOM_ENGINE_DEFAULT            = AF_RANDOM_ENGINE_PHILOX               //Resolves to Philox
} af_random_engine_type;
#endif

////////////////////////////////////////////////////////////////////////////////
// FORGE / Graphics Related Enums
// These enums have values corresponsding to Forge enums in forge defines.h
////////////////////////////////////////////////////////////////////////////////
typedef enum {
    AF_COLORMAP_DEFAULT = 0,    ///< Default grayscale map
    AF_COLORMAP_SPECTRUM= 1,    ///< Spectrum map (390nm-830nm, in sRGB colorspace)
    AF_COLORMAP_COLORS  = 2,    ///< Colors, aka. Rainbow
    AF_COLORMAP_RED     = 3,    ///< Red hue map
    AF_COLORMAP_MOOD    = 4,    ///< Mood map
    AF_COLORMAP_HEAT    = 5,    ///< Heat map
    AF_COLORMAP_BLUE    = 6,    ///< Blue hue map
    AF_COLORMAP_INFERNO = 7,    ///< Perceptually uniform shades of black-red-yellow
    AF_COLORMAP_MAGMA   = 8,    ///< Perceptually uniform shades of black-red-white
    AF_COLORMAP_PLASMA  = 9,    ///< Perceptually uniform shades of blue-red-yellow
    AF_COLORMAP_VIRIDIS = 10    ///< Perceptually uniform shades of blue-green-yellow
} af_colormap;

#if AF_API_VERSION >= 32
typedef enum {
    AF_MARKER_NONE         = 0,
    AF_MARKER_POINT        = 1,
    AF_MARKER_CIRCLE       = 2,
    AF_MARKER_SQUARE       = 3,
    AF_MARKER_TRIANGLE     = 4,
    AF_MARKER_CROSS        = 5,
    AF_MARKER_PLUS         = 6,
    AF_MARKER_STAR         = 7
} af_marker_type;
#endif
////////////////////////////////////////////////////////////////////////////////

#if AF_API_VERSION >= 35
typedef enum {
    AF_CANNY_THRESHOLD_MANUAL    = 0, ///< User has to define canny thresholds manually
    AF_CANNY_THRESHOLD_AUTO_OTSU = 1  ///< Determine canny algorithm thresholds using Otsu algorithm
} af_canny_threshold;
#endif

#if AF_API_VERSION >= 34
typedef enum {
    AF_STORAGE_DENSE     = 0,   ///< Storage type is dense
    AF_STORAGE_CSR       = 1,   ///< Storage type is CSR
    AF_STORAGE_CSC       = 2,   ///< Storage type is CSC
    AF_STORAGE_COO       = 3    ///< Storage type is COO
} af_storage;
#endif

#if AF_API_VERSION >= 36
typedef enum {
    AF_FLUX_QUADRATIC   = 1,    ///< Quadratic flux function
    AF_FLUX_EXPONENTIAL = 2,    ///< Exponential flux function
    AF_FLUX_DEFAULT     = 0     ///< Default flux function is exponential
} af_flux_function;

typedef enum {
    AF_DIFFUSION_GRAD = 1,      ///< Gradient diffusion equation
    AF_DIFFUSION_MCDE = 2,      ///< Modified curvature diffusion equation
    AF_DIFFUSION_DEFAULT = 0    ///< Default option is same as AF_DIFFUSION_GRAD
} af_diffusion_eq;

typedef enum {
    AF_TOPK_MIN     = 1,  ///< Top k min values
    AF_TOPK_MAX     = 2,  ///< Top k max values
    AF_TOPK_DEFAULT = 0   ///< Default option (max)
} af_topk_function;
#endif

#if AF_API_VERSION >= 37
typedef enum {
    AF_VARIANCE_DEFAULT    = 0, ///< Default (Population) variance
    AF_VARIANCE_SAMPLE     = 1, ///< Sample variance
    AF_VARIANCE_POPULATION = 2  ///< Population variance
} af_var_bias;

typedef enum {
    AF_ITERATIVE_DECONV_LANDWEBER       = 1,        ///< Landweber Deconvolution
    AF_ITERATIVE_DECONV_RICHARDSONLUCY  = 2,        ///< Richardson-Lucy Deconvolution
    AF_ITERATIVE_DECONV_DEFAULT         = 0         ///< Default is Landweber deconvolution
} af_iterative_deconv_algo;

typedef enum {
    AF_INVERSE_DECONV_TIKHONOV       = 1,        ///< Tikhonov Inverse deconvolution
    AF_INVERSE_DECONV_DEFAULT        = 0         ///< Default is Tikhonov deconvolution
} af_inverse_deconv_algo;

#endif

#if AF_API_VERSION >= 37
typedef enum {
    AF_CONV_GRADIENT_DEFAULT = 0,
    AF_CONV_GRADIENT_FILTER  = 1,
    AF_CONV_GRADIENT_DATA    = 2,
    AF_CONV_GRADIENT_BIAS    = 3
} af_conv_gradient_type;
#endif

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
#if AF_API_VERSION >= 31
    typedef af_ycc_std YCCStd;
#endif
#if AF_API_VERSION >= 31
    typedef af_image_format imageFormat;
#endif
#if AF_API_VERSION >= 32
    typedef af_backend Backend;
#endif
#if AF_API_VERSION >= 32
    typedef af_marker_type markerType;
#endif
#if AF_API_VERSION >= 34
    typedef af_moment_type momentType;
#endif
#if AF_API_VERSION >= 34
    typedef af_storage storage;
#endif
#if AF_API_VERSION >= 34
    typedef af_binary_op binaryOp;
#endif
#if AF_API_VERSION >= 34
    typedef af_random_engine_type randomEngineType;
#endif
#if AF_API_VERSION >= 35
    typedef af_canny_threshold cannyThreshold;
#endif
#if AF_API_VERSION >= 36
    typedef af_flux_function fluxFunction;
    typedef af_diffusion_eq diffusionEq;
    typedef af_topk_function topkFunction;
#endif
#if AF_API_VERSION >= 37
    typedef af_var_bias varBias;
    typedef af_iterative_deconv_algo iterativeDeconvAlgo;
    typedef af_inverse_deconv_algo inverseDeconvAlgo;
    typedef af_conv_gradient_type convGradientType;
#endif
}

#endif
