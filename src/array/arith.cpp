/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/arith.h>
#include <af/index.h>
#include "error.hpp"

namespace af
{

    ///////////////////////////////////////////////////////////////////////////
    // Operator +=, -=, *=, /=
    ///////////////////////////////////////////////////////////////////////////
#define INSTANTIATE(op, func)                                               \
    array& array::operator op(const array &other)                           \
    {                                                                       \
        return *this = *this op other;                                      \
    }                                                                       \
    array& array::operator op(const double &value)                          \
    {                                                                       \
        array cst = constant(value, this->dims(), this->type());            \
        return *this = *this op cst;                                        \
    }                                                                       \
    array& array::operator op(const af_cdouble &value)                      \
    {                                                                       \
        array cst = constant(value, this->dims());                          \
        return *this = *this op cst;                                        \
    }                                                                       \
    array& array::operator op(const af_cfloat &value)                       \
    {                                                                       \
        array cst = constant(value, this->dims());                          \
        return *this = *this op cst;                                        \
    }                                                                       \

    INSTANTIATE(+=, af_add)
    INSTANTIATE(-=, af_sub)
    INSTANTIATE(*=, af_mul)
    INSTANTIATE(/=, af_div)

#undef INSTANTIATE

    ///////////////////////////////////////////////////////////////////////////
    // Operator +, -, *, /
    ///////////////////////////////////////////////////////////////////////////
#define INSTANTIATE(op, func)                                               \
    array array::operator op(const array &other) const                      \
    {                                                                       \
        af_array out;                                                       \
        AF_THROW(func(&out, this->get(), other.get()));                     \
        return array(out);                                                  \
    }                                                                       \
    array array::operator op(const double &value) const                     \
    {                                                                       \
        af_array out;                                                       \
        array cst = constant(value, this->dims(), this->type());            \
        AF_THROW(func(&out, this->get(), cst.get()));                       \
        return array(out);                                                  \
    }                                                                       \
    array array::operator op(const af_cdouble &value) const                 \
    {                                                                       \
        af_array out;                                                       \
        array cst = constant(value, this->dims());                          \
        AF_THROW(func(&out, this->get(), cst.get()));                       \
        return array(out);                                                  \
    }                                                                       \
    array array::operator op(const af_cfloat &value) const                  \
    {                                                                       \
        af_array out;                                                       \
        array cst = constant(value, this->dims());                          \
        AF_THROW(func(&out, this->get(), cst.get()));                       \
        return array(out);                                                  \
    }                                                                       \
    array operator op(const double &value, const array &other)              \
    {                                                                       \
        af_array out;                                                       \
        array cst = constant(value, other.dims(), other.type());            \
        AF_THROW(func(&out, other.get(), cst.get()));                       \
        return array(out);                                                  \
    }                                                                       \
    array operator op(const af_cdouble &value, const array& other)          \
    {                                                                       \
        af_array out;                                                       \
        array cst = constant(value, other.dims());                          \
        AF_THROW(func(&out, other.get(), cst.get()));                       \
        return array(out);                                                  \
    }                                                                       \
    array operator op(const af_cfloat &value, const array& other)           \
    {                                                                       \
        af_array out;                                                       \
        array cst = constant(value, other.dims());                          \
        AF_THROW(func(&out, other.get(), cst.get()));                       \
        return array(out);                                                  \
    }                                                                       \

    INSTANTIATE(+, af_add)
    INSTANTIATE(-, af_sub)
    INSTANTIATE(*, af_mul)
    INSTANTIATE(/, af_div)

#undef INSTANTIATE

    ///////////////////////////////////////////////////////////////////////////
    // Operator ==, !=, < <=, >, >=
    ///////////////////////////////////////////////////////////////////////////
#define INSTANTIATE(op, func)                                               \
    array array::operator op(const array &other) const                      \
    {                                                                       \
        af_array out;                                                       \
        AF_THROW(func(&out, this->get(), other.get()));                     \
        return array(out);                                                  \
    }                                                                       \
    array array::operator op(const bool &value) const                       \
    {                                                                       \
        af_array out;                                                       \
        array cst = constant(value, this->dims(), this->type());            \
        AF_THROW(func(&out, this->get(), cst.get()));                       \
        return array(out);                                                  \
    }                                                                       \
    array array::operator op(const int &value) const                        \
    {                                                                       \
        af_array out;                                                       \
        array cst = constant(value, this->dims(), this->type());            \
        AF_THROW(func(&out, this->get(), cst.get()));                       \
        return array(out);                                                  \
    }                                                                       \
    array array::operator op(const double &value) const                     \
    {                                                                       \
        af_array out;                                                       \
        array cst = constant(value, this->dims(), this->type());            \
        AF_THROW(func(&out, this->get(), cst.get()));                       \
        return array(out);                                                  \
    }                                                                       \
    array array::operator op(const af_cdouble &value) const                 \
    {                                                                       \
        af_array out;                                                       \
        array cst = constant(value, this->dims());                          \
        AF_THROW(func(&out, this->get(), cst.get()));                       \
        return array(out);                                                  \
    }                                                                       \
    array array::operator op(const af_cfloat &value) const                  \
    {                                                                       \
        af_array out;                                                       \
        array cst = constant(value, this->dims());                          \
        AF_THROW(func(&out, this->get(), cst.get()));                       \
        return array(out);                                                  \
    }                                                                       \
    array operator op(const bool &value, const array &other)                \
    {                                                                       \
        af_array out;                                                       \
        array cst = constant(value, other.dims(), other.type());            \
        AF_THROW(func(&out, other.get(), cst.get()));                       \
        return array(out);                                                  \
    }                                                                       \
    array operator op(const int &value, const array &other)                 \
    {                                                                       \
        af_array out;                                                       \
        array cst = constant(value, other.dims(), other.type());            \
        AF_THROW(func(&out, other.get(), cst.get()));                       \
        return array(out);                                                  \
    }                                                                       \
    array operator op(const double &value, const array &other)              \
    {                                                                       \
        af_array out;                                                       \
        array cst = constant(value, other.dims(), other.type());            \
        AF_THROW(func(&out, other.get(), cst.get()));                       \
        return array(out);                                                  \
    }                                                                       \
    array operator op(const af_cdouble &value, const array& other)          \
    {                                                                       \
        af_array out;                                                       \
        array cst = constant(value, other.dims());                          \
        AF_THROW(func(&out, other.get(), cst.get()));                       \
        return array(out);                                                  \
    }                                                                       \
    array operator op(const af_cfloat &value, const array& other)           \
    {                                                                       \
        af_array out;                                                       \
        array cst = constant(value, other.dims());                          \
        AF_THROW(func(&out, other.get(), cst.get()));                       \
        return array(out);                                                  \
    }                                                                       \

    INSTANTIATE(==, af_eq)
    INSTANTIATE(!=, af_neq)
    INSTANTIATE(< , af_lt)
    INSTANTIATE(<=, af_le)
    INSTANTIATE(> , af_gt)
    INSTANTIATE(>=, af_ge)

#undef INSTANTIATE
}
