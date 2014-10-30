#include <af/array.h>
#include <af/arith.h>
#include <af/index.h>
#include "error.hpp"

namespace af
{
    ///////////////////////////////////////////////////////////////////////////
    // Operator Overloading for array
    ///////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    // Operator =
    ///////////////////////////////////////////////////////////////////////////
    array& array::operator=(const array &other)
    {
        if (isRef) {
            AF_THROW(af_assign(arr, numdims(), s, other.get()));
            isRef = false;
        } else {
            if (this->get() == other.get()) {
                return *this;
            }
            if(this->get() != 0) {
                AF_THROW(af_destroy_array(this->get()));
            }

            af_array temp = 0;
            AF_THROW(af_weak_copy(&temp, other.get()));
            this->arr = temp;
        }
        return *this;
    }

    array& array::operator=(const double &value)
    {
        if (isRef) {
            array cst = constant(value, this->dims(), this->type());
            AF_THROW(af_assign(arr, numdims(), s, cst.get()));
            isRef = false;
        } else {
            if(this->get() != 0) {
                AF_THROW(af_destroy_array(this->get()));
            }
            AF_THROW(af_constant(&arr, value, numdims(), dims().get(), type()));
        }
        return *this;
    }

    array& array::operator=(const af_cdouble &value)
    {
        if (isRef) {
            array cst = constant(value, this->dims());
            AF_THROW(af_assign(arr, numdims(), s, cst.get()));
            isRef = false;
        } else {
            if(this->get() != 0) {
                AF_THROW(af_destroy_array(this->get()));
            }
            AF_THROW(af_constant_c64(&arr, (const void*)&value, numdims(), dims().get()));
        }
        return *this;
    }

    array& array::operator=(const af_cfloat &value)
    {
        if (isRef) {
            array cst = constant(value, this->dims());
            AF_THROW(af_assign(arr, numdims(), s, cst.get()));
            isRef = false;
        } else {
            if(this->get() != 0) {
                AF_THROW(af_destroy_array(this->get()));
            }
            AF_THROW(af_constant_c32(&arr, (const void*)&value, numdims(), dims().get()));
        }
        return *this;
    }

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
