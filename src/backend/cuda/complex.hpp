/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <binary.hpp>
#include <common/jit/UnaryNode.hpp>
#include <optypes.hpp>
#include <af/dim4.hpp>

namespace cuda {
template<typename To, typename Ti>
Array<To> cplx(const Array<Ti> &lhs, const Array<Ti> &rhs,
               const af::dim4 &odims) {
    return createBinaryNode<To, Ti, af_cplx2_t>(lhs, rhs, odims);
}

template<typename To, typename Ti>
Array<To> real(const Array<Ti> &in) {
    common::Node_ptr in_node = in.getNode();
    common::UnaryNode *node  = new common::UnaryNode(
        getFullName<To>(), shortname<To>(true), "__creal", in_node, af_real_t);

    return createNodeArray<To>(in.dims(), common::Node_ptr(node));
}

template<typename To, typename Ti>
Array<To> imag(const Array<Ti> &in) {
    common::Node_ptr in_node = in.getNode();
    common::UnaryNode *node  = new common::UnaryNode(
        getFullName<To>(), shortname<To>(true), "__cimag", in_node, af_imag_t);

    return createNodeArray<To>(in.dims(), common::Node_ptr(node));
}

template<typename T>
static const char *abs_name() {
    return "fabs";
}
template<>
STATIC_ const char *abs_name<cfloat>() {
    return "__cabsf";
}
template<>
STATIC_ const char *abs_name<cdouble>() {
    return "__cabs";
}

template<typename To, typename Ti>
Array<To> abs(const Array<Ti> &in) {
    common::Node_ptr in_node = in.getNode();
    common::UnaryNode *node =
        new common::UnaryNode(getFullName<To>(), shortname<To>(true),
                              abs_name<Ti>(), in_node, af_abs_t);

    return createNodeArray<To>(in.dims(), common::Node_ptr(node));
}

template<typename T>
static const char *conj_name() {
    return "__noop";
}
template<>
STATIC_ const char *conj_name<cfloat>() {
    return "__cconjf";
}
template<>
STATIC_ const char *conj_name<cdouble>() {
    return "__cconj";
}

template<typename T>
Array<T> conj(const Array<T> &in) {
    common::Node_ptr in_node = in.getNode();
    common::UnaryNode *node =
        new common::UnaryNode(getFullName<T>(), shortname<T>(true),
                              conj_name<T>(), in_node, af_conj_t);

    return createNodeArray<T>(in.dims(), common::Node_ptr(node));
}
}  // namespace cuda
