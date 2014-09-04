#pragma once
#include <algorithm>
#include <limits>
#include <cuComplex.h>
#include "complex.hpp"

#ifdef __DH__
#undef __DH__
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <math_functions.h>
#include <math_constants.h>
#define __DH__ __device__ __host__
#else
#define __DH__
#endif

namespace cuda
{
    typedef cuFloatComplex   cfloat;
    typedef cuDoubleComplex cdouble;
    typedef unsigned int   uint;
    typedef unsigned char uchar;

    template<typename T> static inline __DH__ T abs(T val)  { return abs(val); }
    static inline __DH__ float  abs(float  val) { return fabsf(val); }
    static inline __DH__ double abs(double val) { return fabs (val); }
    static inline __DH__ float  abs(cfloat  cval) { return cuCabsf(cval); }
    static inline __DH__ double abs(cdouble cval) { return cuCabs (cval); }

#ifndef __CUDA_ARCH__
    template<typename T> static inline __DH__ T min(T lhs, T rhs) { return std::min(lhs, rhs);}
    template<typename T> static inline __DH__ T max(T lhs, T rhs) { return std::max(lhs, rhs);}
#else
    template<typename T> static inline __DH__ T min(T lhs, T rhs) { return ::min(lhs, rhs);}
    template<typename T> static inline __DH__ T max(T lhs, T rhs) { return ::max(lhs, rhs);}
#endif

    template<> __DH__
    cfloat max<cfloat>(cfloat lhs, cfloat rhs)
    {
        return abs(lhs) > abs(rhs) ? lhs : rhs;
    }

    template<> __DH__
    cdouble max<cdouble>(cdouble lhs, cdouble rhs)
    {
        return abs(lhs) > abs(rhs) ? lhs : rhs;
    }

    template<> __DH__
    cfloat min<cfloat>(cfloat lhs, cfloat rhs)
    {
        return abs(lhs) < abs(rhs) ? lhs :  rhs;
    }

    template<> __DH__
    cdouble min<cdouble>(cdouble lhs, cdouble rhs)
    {
        return abs(lhs) < abs(rhs) ? lhs :  rhs;
    }

    template<typename T>
    static __DH__
    T constant(double val)
    {
        return (T)(val);
    }

    template<> __DH__
    cfloat constant<cfloat >(double val)
    {
        cfloat  cval = {(float)val, 0};
        return cval;
    }

    template<> __DH__
    cdouble constant<cdouble >(double val)
    {
        cdouble  cval = {val, 0};
        return cval;
    }

#ifndef __CUDA_ARCH__
    template <typename T> T limit_max() { return std::numeric_limits<T>::max(); }
    template <typename T> T limit_min() { return std::numeric_limits<T>::min(); }
#else
    template <typename T> __device__ T limit_max() { return 1u << (8 * sizeof(T) - 1); }
    template <typename T> __device__ T limit_min() { return constant<T>(0); }

    template<> __device__  int    limit_max<int>()    { return 0x7fffffff; }
    template<> __device__  int    limit_min<int>()    { return 0x80000000; }
    template<> __device__  char   limit_max<char>()   { return 0x7f; }
    template<> __device__  char   limit_min<char>()   { return 0x80; }
    template<> __device__  float  limit_max<float>()  { return  CUDART_INF_F; }
    template<> __device__  float  limit_min<float>()  { return -CUDART_INF_F; }
    template<> __device__  double limit_max<double>() { return  CUDART_INF_F; }
    template<> __device__  double limit_min<double>() { return -CUDART_INF_F; }
#endif
}

namespace detail = cuda;
