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
#include <string>
#include <sstream>

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
    if (std::isinf(val)) { return val < 0 ? NINF : PINF; }
    return std::to_string(val);
}

template<>
std::string ToNumStr<double>::operator()(double val) {
    static const char *PINF = "+INFINITY";
    static const char *NINF = "-INFINITY";
    if (std::isinf(val)) { return val < 0 ? NINF : PINF; }
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
std::string ToNumStr<common::half>::operator()(common::half val) {
    static const char *PINF = "+INFINITY";
    static const char *NINF = "-INFINITY";
    if (common::isinf(val)) { return val < 0 ? NINF : PINF; }
    return std::to_string(val);
}

template<>
template<>
std::string ToNumStr<common::half>::operator()<float>(float val) {
    static const char *PINF = "+INFINITY";
    static const char *NINF = "-INFINITY";
    if (common::isinf(common::half(val))) { return val < 0 ? NINF : PINF; }
    return std::to_string(val);
}


#define INSTANTIATE(TYPE)                       \
  template struct ToNumStr<TYPE>

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
  INSTANTIATE(common::half);

#undef INSTANTIATE

}  // namespace opencl

namespace common {
template<typename T>
class kernel_type;
}

namespace common {
template<>
struct kernel_type<common::half> {
    using data = common::half;

    using compute = cl_half;

    // These are the types within a kernel
    using native = cl_half;
};
}
