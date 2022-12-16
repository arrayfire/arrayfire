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
#include <common/util.hpp>
#include <type_util.hpp>

#include <cmath>
#include <sstream>
#include <string>

using arrayfire::common::half;
using arrayfire::common::toString;

using std::isinf;
using std::stringstream;

namespace arrayfire {
namespace opencl {

template<typename T>
inline std::string ToNumStr<T>::operator()(T val) {
    ToNum<T> toNum;
    return toString(toNum(val));
}

template<>
std::string ToNumStr<float>::operator()(float val) {
    static const char *PINF = "+INFINITY";
    static const char *NINF = "-INFINITY";
    if (isinf(val)) { return val < 0.f ? NINF : PINF; }
    return toString(val);
}

template<>
std::string ToNumStr<double>::operator()(double val) {
    static const char *PINF = "+INFINITY";
    static const char *NINF = "-INFINITY";
    if (isinf(val)) { return val < 0. ? NINF : PINF; }
    return toString(val);
}

template<>
std::string ToNumStr<cfloat>::operator()(cfloat val) {
    ToNumStr<float> realStr;
    stringstream s;
    s << "{" << realStr(val.s[0]) << "," << realStr(val.s[1]) << "}";
    return s.str();
}

template<>
std::string ToNumStr<cdouble>::operator()(cdouble val) {
    ToNumStr<double> realStr;
    stringstream s;
    s << "{" << realStr(val.s[0]) << "," << realStr(val.s[1]) << "}";
    return s.str();
}

template<>
std::string ToNumStr<half>::operator()(half val) {
    using namespace std;
    using namespace common;
    static const char *PINF = "+INFINITY";
    static const char *NINF = "-INFINITY";
    if (isinf(val)) { return val < 0.f ? NINF : PINF; }
    return toString(val);
}

template<>
template<>
std::string ToNumStr<half>::operator()<float>(float val) {
    static const char *PINF = "+INFINITY";
    static const char *NINF = "-INFINITY";
    if (isinf(half(val))) { return val < 0.f ? NINF : PINF; }
    return toString(val);
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
}  // namespace arrayfire
