/*******************************************************
 * Copyright (c) 2023, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <CL/sycl.hpp>
#include <Param.hpp>
#include <common/dispatch.hpp>
#include <common/half.hpp>
#include <common/traits.hpp>
#include <debug_oneapi.hpp>
#include <traits.hpp>

#include <algorithm>
#include <complex>
#include <string>
#include <vector>

using arrayfire::common::half;

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename inType, typename outType>
outType convertType(inType value) {
    return static_cast<outType>(value);
}

template<>
char convertType<compute_t<half>, char>(compute_t<half> value) {
    return (char)((short)value);
}

template<>
compute_t<half> convertType<char, compute_t<half>>(char value) {
    return compute_t<half>(value);
}

template<>
unsigned char convertType<compute_t<half>, unsigned char>(
    compute_t<half> value) {
    return (unsigned char)((short)value);
}

template<>
compute_t<half> convertType<unsigned char, compute_t<half>>(
    unsigned char value) {
    return compute_t<half>(value);
}

template<>
cdouble convertType<cfloat, cdouble>(cfloat value) {
    return cdouble(value.real(), value.imag());
}

template<>
cfloat convertType<cdouble, cfloat>(cdouble value) {
    return cfloat(value.real(), value.imag());
}

template<typename T>
T scale(T value, double factor) {
    return (T)(double(value) * factor);
}

template<>
cfloat scale<cfloat>(cfloat value, double factor) {
    return cfloat{static_cast<float>(value.real() * factor),
                  static_cast<float>(value.imag() * factor)};
}

template<>
cdouble scale<cdouble>(cdouble value, double factor) {
    return cdouble{value.real() * factor, value.imag() * factor};
}

#define INSTANTIATE_SCALE(T) template T scale<T>(T value, double factor)

INSTANTIATE_SCALE(float);
INSTANTIATE_SCALE(double);
INSTANTIATE_SCALE(int);
INSTANTIATE_SCALE(uint);
INSTANTIATE_SCALE(intl);
INSTANTIATE_SCALE(uintl);
INSTANTIATE_SCALE(short);
INSTANTIATE_SCALE(ushort);
INSTANTIATE_SCALE(uchar);
INSTANTIATE_SCALE(char);
INSTANTIATE_SCALE(half);

#define INSTANTIATE_DESTINATIONS(SRC_T, DST_T) \
    template DST_T convertType<SRC_T, DST_T>(SRC_T value)

#define INSTANTIATE_SOURCES(SRC_T)           \
    INSTANTIATE_DESTINATIONS(SRC_T, float);  \
    INSTANTIATE_DESTINATIONS(SRC_T, double); \
    INSTANTIATE_DESTINATIONS(SRC_T, int);    \
    INSTANTIATE_DESTINATIONS(SRC_T, uint);   \
    INSTANTIATE_DESTINATIONS(SRC_T, intl);   \
    INSTANTIATE_DESTINATIONS(SRC_T, uintl);  \
    INSTANTIATE_DESTINATIONS(SRC_T, short);  \
    INSTANTIATE_DESTINATIONS(SRC_T, ushort); \
    INSTANTIATE_DESTINATIONS(SRC_T, uchar);  \
    INSTANTIATE_DESTINATIONS(SRC_T, char);   \
    INSTANTIATE_DESTINATIONS(SRC_T, half)

INSTANTIATE_SOURCES(float);
INSTANTIATE_SOURCES(double);
INSTANTIATE_SOURCES(int);
INSTANTIATE_SOURCES(uint);
INSTANTIATE_SOURCES(intl);
INSTANTIATE_SOURCES(uintl);
INSTANTIATE_SOURCES(short);
INSTANTIATE_SOURCES(ushort);
INSTANTIATE_SOURCES(uchar);
INSTANTIATE_SOURCES(char);
INSTANTIATE_SOURCES(half);

INSTANTIATE_DESTINATIONS(std::complex<float>, std::complex<float>);
INSTANTIATE_DESTINATIONS(std::complex<float>, std::complex<double>);
INSTANTIATE_DESTINATIONS(std::complex<double>, std::complex<float>);
INSTANTIATE_DESTINATIONS(std::complex<double>, std::complex<double>);

#undef INSTANTIATE_SOURCES
#undef INSTANTIATE_DESTINATIONS

#define OTHER_SPECIALIZATIONS(IN_T)                      \
    template<>                                           \
    cfloat convertType<IN_T, cfloat>(IN_T value) {       \
        return cfloat(static_cast<float>(value), 0.0f);  \
    }                                                    \
                                                         \
    template<>                                           \
    cdouble convertType<IN_T, cdouble>(IN_T value) {     \
        return cdouble(static_cast<double>(value), 0.0); \
    }

OTHER_SPECIALIZATIONS(float)
OTHER_SPECIALIZATIONS(double)
OTHER_SPECIALIZATIONS(int)
OTHER_SPECIALIZATIONS(uint)
OTHER_SPECIALIZATIONS(intl)
OTHER_SPECIALIZATIONS(uintl)
OTHER_SPECIALIZATIONS(short)
OTHER_SPECIALIZATIONS(ushort)
OTHER_SPECIALIZATIONS(uchar)
OTHER_SPECIALIZATIONS(char)
OTHER_SPECIALIZATIONS(half)

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
