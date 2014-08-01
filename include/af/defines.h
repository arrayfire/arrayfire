#pragma once

typedef enum {
    AF_SUCCESS=0,
    AF_ERR_INTERNAL,
    AF_ERR_NOMEM,
    AF_ERR_DRIVER,
    AF_ERR_RUNTIME,
    AF_ERR_INVALID_ARRAY,
    AF_ERR_ARG,
    AF_ERR_SIZE,
    AF_ERR_NOT_SUPPORTED,
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

typedef long dim_type;

#include <cstddef>
typedef struct af_seq {
    size_t begin, end;
    int    step;
} af_seq;
static const af_seq span = {1, 1, 0};

#ifdef __cplusplus
#include <complex>
#ifdef AF_CUDA
#include <cuComplex.h>
#include <vector_types.h>
typedef cuFloatComplex cfloat;
typedef cuDoubleComplex cdouble;
#else
typedef std::complex<float> cuFloatComplex;
typedef std::complex<double> cuDoubleComplex;
typedef std::complex<float> cfloat;
typedef std::complex<double> cdouble;
//typedef char[4]             char4;
//typedef unsigned char[4]    uchar4;
#endif

namespace af {

    typedef af_dtype dtype;

    //TODO: Move to seperate file
    template<typename T> struct dtype_traits;
    template<>
    struct dtype_traits<float> {
        enum { af_type = f32 };
        typedef float cuda_type;
        typedef float* cuda_type_ptr;
        static const char* getName() { return "float"; }
    };
    template<>
    struct dtype_traits<cfloat> {
        enum { af_type = c32 };
        typedef cuFloatComplex cuda_type;
        typedef cuda_type* cuda_type_ptr;
        static const char* getName() { return "cfloat"; }
    };
    template<>
    struct dtype_traits<double> {
        enum { af_type = f64 };
        typedef double  cuda_type;
        typedef cuda_type* cuda_type_ptr;
        static const char* getName() { return "double"; }
    };
    template<>
    struct dtype_traits<cdouble> {
        enum { af_type = c64 };
        typedef cuDoubleComplex  cuda_type;
        typedef cuda_type* cuda_type_ptr;
        static const char* getName() { return "cdouble"; }
    };
    template<>
    struct dtype_traits<char> {
        enum { af_type = b8 };
        typedef bool  cuda_type;
        typedef cuda_type* cuda_type_ptr;
        static const char* getName() { return "char"; }
    };
    template<>
    struct dtype_traits<int> {
        enum { af_type = s32 };
        typedef int  cuda_type;
        typedef cuda_type* cuda_type_ptr;
        static const char* getName() { return "int"; }
    };
    template<>
    struct dtype_traits<unsigned> {
        enum { af_type = u32 };
        typedef unsigned  cuda_type;
        typedef cuda_type* cuda_type_ptr;
        static const char* getName() { return "unsigned"; }
    };
    template<>
    struct dtype_traits<unsigned char> {
        enum { af_type = u8 };
        typedef unsigned  cuda_type;
        typedef cuda_type* cuda_type_ptr;
        static const char* getName() { return "unsigned char"; }
    };
// TODO: Add combined types
//    template<>
//    struct dtype_traits<uchar4> {
//        enum { af_type = u8x4 };
//        typedef unsigned  cuda_type;
//        typedef cuda_type* cuda_type_ptr;
//    };
//    template<>
//    struct dtype_traits<char4> {
//        enum { af_type = s8x4 };
//        typedef unsigned  cuda_type;
//        typedef cuda_type* cuda_type_ptr;
//    };
}
#endif
