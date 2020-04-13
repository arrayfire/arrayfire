/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <types.hpp>

#include <common/half.hpp>
#include <type_util.hpp>

#include <cmath>
#include <sstream>
#include <string>

using common::half;
using common::to_string;
using std::to_string;
using std::move;

namespace opencl {

template<typename T>
inline std::string ToNumStr<T>::operator()(T val) {
    ToNum<T> toNum;
    return std::to_string(toNum(val));
}

template<>
std::string ToNumStr<float>::operator()(float val) {
    static const char *PINF = "+INFINITY";
    static const char *NINF = "-INFINITY";
    if (std::isinf(val)) { return val < 0.f ? NINF : PINF; }
    return std::to_string(val);
}

template<>
std::string ToNumStr<double>::operator()(double val) {
    static const char *PINF = "+INFINITY";
    static const char *NINF = "-INFINITY";
    if (std::isinf(val)) { return val < 0. ? NINF : PINF; }
    return std::to_string(val);
}

template<>
std::string ToNumStr<cfloat>::operator()(cfloat val) {
    ToNumStr<float> realStr;
    std::stringstream s;
    s << "{" << realStr(val.s[0]) << "," << realStr(val.s[1]) << "}";
    return s.str();
}

template<>
std::string ToNumStr<cdouble>::operator()(cdouble val) {
    ToNumStr<double> realStr;
    std::stringstream s;
    s << "{" << realStr(val.s[0]) << "," << realStr(val.s[1]) << "}";
    return s.str();
}

template<>
std::string ToNumStr<half>::operator()(half val) {
    using namespace std;
    using namespace common;
    static const char *PINF = "+INFINITY";
    static const char *NINF = "-INFINITY";
    if (common::isinf(val)) { return val < 0.f ? NINF : PINF; }
    return to_string(val);
}

template<>
template<>
std::string ToNumStr<half>::operator()<float>(float val) {
    static const char *PINF = "+INFINITY";
    static const char *NINF = "-INFINITY";
    if (common::isinf(half(val))) { return val < 0.f ? NINF : PINF; }
    return std::to_string(val);
}

#define INSTANTIATE(TYPE) template struct ToNumStr<TYPE>

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(cfloat);
INSTANTIATE(cdouble);
INSTANTIATE(short);
INSTANTIATE(ushort);
INSTANTIATE(int);
INSTANTIATE(uint);
INSTANTIATE(intl);
INSTANTIATE(uintl);
INSTANTIATE(uchar);
INSTANTIATE(char);
INSTANTIATE(half);

#undef INSTANTIATE

}  // namespace opencl
