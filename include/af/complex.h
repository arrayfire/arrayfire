/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include "af/defines.h"


#ifdef __cplusplus
#include <ostream>
#include <istream>

namespace af{
#endif

#ifdef __cplusplus
extern "C" {
#endif
typedef struct af_cfloat {
    float real;
    float imag;
#ifdef __cplusplus
    af_cfloat(const float real = 0, const float imag = 0) :real(real), imag(imag) {};
#endif
} af_cfloat;

typedef struct af_cdouble {
    double real;
    double imag;
#ifdef __cplusplus
    af_cdouble(const double real = 0, const double imag = 0) :real(real), imag(imag) {}
#endif
} af_cdouble;
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
typedef af::af_cfloat   cfloat;
typedef af::af_cdouble  cdouble;

AFAPI float real(af_cfloat val);
AFAPI double real(af_cdouble val);

AFAPI float imag(af_cfloat val);
AFAPI double imag(af_cdouble val);

#define DEFINE_OP(OP)                                                   \
    AFAPI af::cfloat  operator OP(const af::cfloat  &lhs, const af::cfloat  &rhs); \
    AFAPI af::cdouble operator OP(const af::cdouble &lhs, const af::cdouble &rhs); \
    AFAPI af::cfloat  operator OP(const af::cfloat  &lhs, const     double  &rhs); \
    AFAPI af::cdouble operator OP(const af::cdouble &lhs, const     double  &rhs); \
    AFAPI af::cfloat  operator OP(const double      &rhs, const af::cfloat  &lhs); \
    AFAPI af::cdouble operator OP(const double      &rhs, const af::cdouble &lhs); \
    AFAPI af::cdouble operator OP(const af::cfloat  &lhs, const af::cdouble &rhs); \
    AFAPI af::cdouble operator OP(const af::cdouble &lhs, const af::cfloat  &rhs); \

DEFINE_OP(+)
DEFINE_OP(-)
DEFINE_OP(*)
DEFINE_OP(/)

#undef DEFINE_OP

AFAPI bool operator==(const cfloat &lhs, const cfloat &rhs);
AFAPI bool operator==(const cdouble &lhs, const cdouble &rhs);

AFAPI bool operator!=(const cfloat &lhs, const cfloat &rhs);
AFAPI bool operator!=(const cdouble &lhs, const cdouble &rhs);

AFAPI std::istream& operator>> (std::istream &is, cfloat &in);
AFAPI std::istream& operator>> (std::istream &is, cdouble &in);

AFAPI std::ostream& operator<< (std::ostream &os, const cfloat &in);
AFAPI std::ostream& operator<< (std::ostream &os, const cdouble &in);


AFAPI float abs(const cfloat &val);
AFAPI double abs(const cdouble &val);

AFAPI cfloat conj(const cfloat &val);
AFAPI cdouble conj(const cdouble &val);

}
#endif
