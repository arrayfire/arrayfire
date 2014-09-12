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

typedef struct af_seq {
    size_t begin, end;
    int    step;
} af_seq;
static const af_seq span = {1, 1, 0};

typedef enum {
    AF_INTERP_NEAREST,
    AF_INTERP_LINEAR,
    AF_INTERP_BILINEAR,
    AF_INTERP_CUBIC
} af_interp_type;
