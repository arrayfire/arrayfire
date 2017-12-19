/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>
#include <common/defines.hpp>
#include "backend.hpp"
#include "types.hpp"

#ifdef __CUDACC__
#include <math_functions.h>
#include <math_constants.h>
#endif

#include <limits>
#include <algorithm>

namespace cuda
{
    template<typename T> static inline __DH__ T abs(T val)  { return abs(val); }
    static inline __DH__ int  abs(int  val) { return (val>0? val : -val); }
    static inline __DH__ char  abs(char  val) { return (val>0? val : -val); }
    static inline __DH__ float  abs(float  val) { return fabsf(val); }
    static inline __DH__ double abs(double val) { return fabs (val); }
    static inline __DH__ float  abs(cfloat  cval) { return cuCabsf(cval); }
    static inline __DH__ double abs(cdouble cval) { return cuCabs (cval); }

    static inline __DH__ size_t min(size_t lhs, size_t rhs) { return lhs < rhs ? lhs : rhs; }
    static inline __DH__ size_t max(size_t lhs, size_t rhs) { return lhs > rhs ? lhs : rhs; }

#ifndef __CUDA_ARCH__
    template<typename T> static inline __DH__ T min(T lhs, T rhs) { return std::min(lhs, rhs);}
    template<typename T> static inline __DH__ T max(T lhs, T rhs) { return std::max(lhs, rhs);}
#else
    template<typename T> static inline __DH__ T min(T lhs, T rhs) { return ::min(lhs, rhs);}
    template<typename T> static inline __DH__ T max(T lhs, T rhs) { return ::max(lhs, rhs);}
#endif

    template<> __DH__
    STATIC_ cfloat  max<cfloat >(cfloat lhs, cfloat rhs)
    {
        return abs(lhs) > abs(rhs) ? lhs : rhs;
    }

    template<> __DH__
    STATIC_ cdouble max<cdouble>(cdouble lhs, cdouble rhs)
    {
        return abs(lhs) > abs(rhs) ? lhs : rhs;
    }

    template<> __DH__
    STATIC_ cfloat  min<cfloat >(cfloat lhs, cfloat rhs)
    {
        return abs(lhs) < abs(rhs) ? lhs :  rhs;
    }

    template<> __DH__
    STATIC_ cdouble min<cdouble>(cdouble lhs, cdouble rhs)
    {
        return abs(lhs) < abs(rhs) ? lhs :  rhs;
    }

    template<typename T> __DH__
    static T scalar(double val)
    {
        return (T)(val);
    }

    template<> __DH__
    STATIC_ cfloat scalar<cfloat >(double val)
    {
        cfloat  cval = {(float)val, 0};
        return cval;
    }

    template<> __DH__
    STATIC_ cdouble scalar<cdouble >(double val)
    {
        cdouble  cval = {val, 0};
        return cval;
    }

    template<typename To, typename Ti> __DH__
    static To scalar(Ti real, Ti imag)
    {
        To  cval = {real, imag};
        return cval;
    }

#ifndef __CUDA_ARCH__
    template <typename T> STATIC_ T      maxval() { return  std::numeric_limits<T     >::max();      }
    template <typename T> STATIC_ T      minval() { return  std::numeric_limits<T     >::min();      }
    template <>           STATIC_ float  maxval() { return  std::numeric_limits<float >::infinity(); }
    template <>           STATIC_ double maxval() { return  std::numeric_limits<double>::infinity(); }
    template <>           STATIC_ float  minval() { return -std::numeric_limits<float >::infinity(); }
    template <>           STATIC_ double minval() { return -std::numeric_limits<double>::infinity(); }
#else
    template <typename T> __device__ T maxval() { return 1u << (8 * sizeof(T) - 1); }
    template <typename T> __device__ T minval() { return scalar<T>(0); }

    template<> __device__  int    maxval<int>()    { return 0x7fffffff; }
    template<> __device__  int    minval<int>()    { return 0x80000000; }
    template<> __device__  intl   maxval<intl>()   { return 0x7fffffffffffffff; }
    template<> __device__  intl   minval<intl>()   { return 0x8000000000000000; }
    template<> __device__  uintl  maxval<uintl>()  { return 1ULL << (8 * sizeof(uintl) - 1); }
    template<> __device__  char   maxval<char>()   { return 0x7f; }
    template<> __device__  char   minval<char>()   { return 0x80; }
    template<> __device__  float  maxval<float>()  { return  CUDART_INF_F; }
    template<> __device__  float  minval<float>()  { return -CUDART_INF_F; }
    template<> __device__  double maxval<double>() { return  CUDART_INF; }
    template<> __device__  double minval<double>() { return -CUDART_INF; }
    template<> __device__  short  maxval<short>()  { return 0x7fff; }
    template<> __device__  short  minval<short>()  { return 0x8000; }
    template<> __device__  ushort maxval<ushort>() { return ((ushort)1) << (8 * sizeof(ushort) - 1); }
#endif

#define upcast cuComplexFloatToDouble
#define downcast cuComplexDoubleToFloat

#ifdef __GNUC__
//This suprresses unused function warnings in gcc
//FIXME: Check if the warnings exist in other compilers
#define __SDH__ static __DH__ __attribute__((unused))
#else
#define __SDH__ static __DH__
#endif
__SDH__ float  real(cfloat  c) { return cuCrealf(c); }
__SDH__ double real(cdouble c) { return cuCreal(c);  }

__SDH__ float  imag(cfloat  c) { return cuCimagf(c); }
__SDH__ double imag(cdouble c) { return cuCimag(c);  }

template<typename T> T
__SDH__  conj(T  x) { return x; }
__SDH__ cfloat  conj(cfloat  c) { return cuConjf(c);}
__SDH__ cdouble conj(cdouble c) { return cuConj(c); }

__SDH__ cfloat make_cfloat(bool     x) { return make_cuComplex(static_cast<float>(x),0);     }
__SDH__ cfloat make_cfloat(int      x) { return make_cuComplex(static_cast<float>(x),0);     }
__SDH__ cfloat make_cfloat(unsigned x) { return make_cuComplex(static_cast<float>(x),0);     }
__SDH__ cfloat make_cfloat(short    x) { return make_cuComplex(static_cast<float>(x),0);     }
__SDH__ cfloat make_cfloat(ushort   x) { return make_cuComplex(static_cast<float>(x),0);     }
__SDH__ cfloat make_cfloat(float    x) { return make_cuComplex(static_cast<float>(x),0);     }
  __SDH__ cfloat make_cfloat(double   x) { return make_cuComplex(static_cast<float>(x),0);     }
  __SDH__ cfloat make_cfloat(cfloat   x) { return x;                    }
  __SDH__ cfloat make_cfloat(cdouble  c) { return make_cuComplex(c.x,c.y); }

__SDH__ cdouble make_cdouble(bool      x) { return make_cuDoubleComplex(static_cast<double>(x),0);       }
__SDH__ cdouble make_cdouble(int       x) { return make_cuDoubleComplex(static_cast<double>(x),0);       }
__SDH__ cdouble make_cdouble(unsigned  x) { return make_cuDoubleComplex(static_cast<double>(x),0);       }
__SDH__ cdouble make_cdouble(short     x) { return make_cuDoubleComplex(static_cast<double>(x),0);       }
__SDH__ cdouble make_cdouble(ushort    x) { return make_cuDoubleComplex(static_cast<double>(x),0);       }
__SDH__ cdouble make_cdouble(float     x) { return make_cuDoubleComplex(static_cast<double>(x),0);       }
__SDH__ cdouble make_cdouble(double    x) { return make_cuDoubleComplex(static_cast<double>(x),0);       }
__SDH__ cdouble make_cdouble(cdouble   x) { return x;                       }
__SDH__ cdouble make_cdouble(cfloat    c) { return make_cuDoubleComplex(static_cast<double>(c.x),c.y);   }

__SDH__ cfloat make_cfloat(float x, float y) { return make_cuComplex(x, y); }
__SDH__ cdouble make_cdouble(double x, double y) { return make_cuDoubleComplex(x, y); }


#define BINOP(OP, cfn, zfn)                                             \
    __SDH__ cfloat   operator OP(cfloat  a, cfloat  b)                  \
    { return cfn(a,b); }                                                \
    __SDH__ cdouble  operator OP(cdouble a, cfloat  b)                  \
    { return zfn(a,upcast(b)); }                                        \
    __SDH__ cdouble  operator OP(cfloat  a, cdouble b)                  \
    { return zfn(upcast(a),b); }                                        \
    __SDH__ cdouble  operator OP(cdouble a, cdouble b)                  \
    { return zfn(a,b); }                                                \
                                                                        \

    BINOP(+, cuCaddf, cuCadd)
    BINOP(-, cuCsubf, cuCsub)
    BINOP(*, cuCmulf, cuCmul)
    BINOP(/, cuCdivf, cuCdiv)

#undef BINOP

#define BINOP_SCALAR(T, TR, R)                  \
    __SDH__ R operator *(TR a, T b)             \
    { return make_##R(a * b.x, a * b.y); }      \
                                                \
    __SDH__ R operator *(T a, TR b)             \
    { return make_##R(a.x * b,  a.y * b); }     \
                                                \
    __SDH__ R operator +(TR a, T b)             \
    { return make_##R(a + b.x, a + b.y); }      \
                                                \
    __SDH__ R operator +(T a, TR b)             \
    { return make_##R(a.x + b,  a.y + b); }     \
                                                \
    __SDH__ R operator -(TR a, T b)             \
    { return make_##R(a - b.x, a - b.y); }      \
                                                \
    __SDH__ R operator -(T a, TR b)             \
    { return make_##R(a.x - b,  a.y - b); }     \
                                                \
    __SDH__ R operator /(T a, TR b)             \
    { return make_##R(a.x / b, a.y / b); }      \
                                                \
    __SDH__ R operator /(TR a, T b)             \
    { return make_##R(a) / b; }                 \
                                                \

    BINOP_SCALAR(cfloat, float, cfloat)
    BINOP_SCALAR(cfloat, double, cdouble)
    BINOP_SCALAR(cdouble, float, cdouble)
    BINOP_SCALAR(cdouble, double, cdouble)

#undef BINOP_SCALAR

__SDH__ bool operator ==(cfloat a, cfloat b) { return (a.x == b.x) && (a.y == b.y); }
__SDH__ bool operator !=(cfloat a, cfloat b) { return !(a == b); }
__SDH__ bool operator ==(cdouble a, cdouble b) { return (a.x == b.x) && (a.y == b.y); }
__SDH__ bool operator !=(cdouble a, cdouble b) { return !(a == b); }

    template<typename T> static inline T division(T lhs, double rhs) { return lhs / rhs; }
    cfloat division(cfloat lhs, double rhs);
    cdouble division(cdouble lhs, double rhs);

    template<typename T>
    static inline __DH__
    T clamp(const T value, const T lo, const T hi)
    {
        return max(lo, min(value, hi));
    }
}
