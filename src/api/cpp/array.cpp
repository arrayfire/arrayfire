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
#include <af/blas.h>
#include <af/data.h>
#include <af/traits.hpp>
#include <af/util.h>
#include <af/index.h>
#include <af/device.h>
#include "error.hpp"

namespace af
{

    static unsigned size_of(af_dtype type)
    {
        switch(type) {
        case f32: return sizeof(float);
        case f64: return sizeof(double);
        case s32: return sizeof(int);
        case u32: return sizeof(unsigned);
        case u8 : return sizeof(unsigned char);
        case b8 : return sizeof(unsigned char);
        case c32: return sizeof(float) * 2;
        case c64: return sizeof(double) * 2;
        default: return sizeof(float);
        }
    }
    array::array() : arr(0), isRef(false) {}
    array::array(const af_array handle): arr(handle), isRef(false) {}

    static void initEmptyArray(af_array *arr, af_dtype ty,
                               dim_type d0, dim_type d1=1, dim_type d2=1, dim_type d3=1)
    {
        dim_type my_dims[] = {d0, d1, d2, d3};
        AF_THROW(af_create_handle(arr, 4, my_dims, ty));
    }

    template<typename T>
    static void initDataArray(af_array *arr, const T *ptr, af_source_t src,
                              dim_type d0, dim_type d1=1, dim_type d2=1, dim_type d3=1)
    {
        af_dtype ty = (af_dtype)dtype_traits<T>::af_type;
        dim_type my_dims[] = {d0, d1, d2, d3};
        switch (src) {
        case afHost:   AF_THROW(af_create_array(arr, (const void * const)ptr, 4, my_dims, ty)); break;
        case afDevice: AF_THROW(af_device_array(arr, (const void *      )ptr, 4, my_dims, ty)); break;
        default: AF_THROW(AF_ERR_INVALID_ARG);
        }
    }

    array::array(const dim4 &dims, af_dtype ty) : arr(0), isRef(false)
    {
        initEmptyArray(&arr, ty, dims[0], dims[1], dims[2], dims[3]);
    }

    array::array(dim_type d0, af_dtype ty) : arr(0), isRef(false)
    {
        initEmptyArray(&arr, ty, d0);
    }

    array::array(dim_type d0, dim_type d1, af_dtype ty) : arr(0), isRef(false)
    {
        initEmptyArray(&arr, ty, d0, d1);
    }

    array::array(dim_type d0, dim_type d1, dim_type d2, af_dtype ty) : arr(0), isRef(false)
    {
        initEmptyArray(&arr, ty, d0, d1, d2);
    }

    array::array(dim_type d0, dim_type d1, dim_type d2, dim_type d3, af_dtype ty) : arr(0), isRef(false)
    {
        initEmptyArray(&arr, ty, d0, d1, d2, d3);
    }

#define INSTANTIATE(T)                                                  \
    template<> AFAPI                                                    \
    array::array(const dim4 &dims, const T *ptr, af_source_t src, dim_type ngfor) \
        : arr(0), isRef(false)                                          \
    {                                                                   \
        initDataArray<T>(&arr, ptr, src, dims[0], dims[1], dims[2], dims[3]); \
    }                                                                   \
    template<> AFAPI                                                    \
    array::array(dim_type d0, const T *ptr, af_source_t src, dim_type ngfor) \
        : arr(0), isRef(false)                                          \
    {                                                                   \
        initDataArray<T>(&arr, ptr, src, d0);                           \
    }                                                                   \
    template<> AFAPI                                                    \
    array::array(dim_type d0, dim_type d1, const T *ptr, af_source_t src, \
                 dim_type ngfor) : arr(0), isRef(false)                 \
    {                                                                   \
        initDataArray<T>(&arr, ptr, src, d0, d1);                       \
    }                                                                   \
    template<> AFAPI                                                    \
    array::array(dim_type d0, dim_type d1, dim_type d2, const T *ptr,   \
                 af_source_t src, dim_type ngfor) : arr(0), isRef(false) \
    {                                                                   \
        initDataArray<T>(&arr, ptr, src, d0, d1, d2);                   \
    }                                                                   \
    template<> AFAPI                                                    \
    array::array(dim_type d0, dim_type d1, dim_type d2, dim_type d3, const T *ptr, \
                 af_source_t src, dim_type ngfor) : arr(0), isRef(false) \
    {                                                                   \
        initDataArray<T>(&arr, ptr, src, d0, d1, d2, d3);               \
    }                                                                   \

    INSTANTIATE(cdouble)
    INSTANTIATE(cfloat)
    INSTANTIATE(double)
    INSTANTIATE(float)
    INSTANTIATE(unsigned)
    INSTANTIATE(int)
    INSTANTIATE(unsigned char)
    INSTANTIATE(char)
#undef INSTANTIATE

    array::~array()
    {
        if (arr) AF_THROW(af_destroy_array(arr));
    }

    af_dtype array::type() const
    {
        af_dtype my_type;
        AF_THROW(af_get_type(&my_type, arr));
        return my_type;
    }

    dim_type array::elements() const
    {
        dim_type elems;
        AF_THROW(af_get_elements(&elems, arr));
        return elems;
    }

    void array::host(void *data) const
    {
        AF_THROW(af_get_data_ptr(data, arr));
    }

    af_array array::get()
    {
        if (!isRef)
            return arr;
        af_array temp = 0;
        af_seq afs[4];
        getSeq(afs);
        AF_THROW(af_index(&temp, arr, 4, afs));
        AF_THROW(af_destroy_array(arr));
        arr = temp;
        isRef = false;
        return arr;
    }

    af_array array::get() const
    {
        return ((array *)(this))->get();
    }

    // Helper functions
    dim4 array::dims() const
    {
        dim_type d0, d1, d2, d3;
        AF_THROW(af_get_dims(&d0, &d1, &d2, &d3, arr));
        return dim4(d0, d1, d2, d3);
    }

    dim_type array::dims(unsigned dim) const
    {
        return dims()[dim];
    }

    unsigned array::numdims() const
    {
        unsigned nd;
        AF_THROW(af_get_numdims(&nd, arr));
        return nd;
    }

    size_t array::bytes() const
    {
        dim_type nElements;
        AF_THROW(af_get_elements(&nElements, arr));
        return nElements * size_of(type());
    }

    array array::copy() const
    {
        af_array other = 0;
        AF_THROW(af_copy_array(&other, arr));
        return array(other);
    }

#undef INSTANTIATE
#define INSTANTIATE(fn)                         \
    bool array::is##fn() const                  \
    {                                           \
        bool ret = false;                       \
        AF_THROW(af_is_##fn(&ret, arr));        \
        return ret;                             \
    }

    INSTANTIATE(empty)
    INSTANTIATE(scalar)
    INSTANTIATE(vector)
    INSTANTIATE(row)
    INSTANTIATE(column)
    INSTANTIATE(complex)
    INSTANTIATE(double)
    INSTANTIATE(single)
    INSTANTIATE(realfloating)
    INSTANTIATE(floating)
    INSTANTIATE(integer)

#undef INSTANTIATE

    array::array(af_array in, seq *seqs) : arr(in), isRef(true)
    {
        for(int i=0; i<4; ++i) s[i] = seqs[i];
    }

    void array::getSeq(af_seq* afs)
    {
        afs[0] = this->s[0].s;
        afs[1] = this->s[1].s;
        afs[2] = this->s[2].s;
        afs[3] = this->s[3].s;
    }

    array array::operator()(const array& idx, unsigned dim) const
    {
        af_array out = 0;
        AF_THROW(af_array_index(&out, this->get(), idx.get(), dim));
        return array(out);
    }

    array array::operator()(const seq &s0) const
    {
        af_array out = 0;
        seq indices[] = {s0, span, span, span};
        //FIXME: check if this->s has same dimensions as numdims
        AF_THROW(af_weak_copy(&out, this->get()));
        return array(out, indices);
    }

    array array::operator()(const seq &s0, const seq &s1) const
    {
        eval();
        af_array out = 0;
        seq indices[] = {s0, s1, span, span};
        //FIXME: check if this->s has same dimensions as numdims
        AF_THROW(af_weak_copy(&out, this->get()));
        return array(out, indices);
    }

    array array::operator()(const seq &s0, const seq &s1, const seq &s3) const
    {
        eval();
        af_array out = 0;
        seq indices[] = {s0, s1, s3, span};
        //FIXME: check if this->s has same dimensions as numdims
        AF_THROW(af_weak_copy(&out, this->get()));
        return array(out, indices);
    }

    array array::operator()(const seq &s0, const seq &s1, const seq &s2, const seq &s3) const
    {
        eval();
        af_array out = 0;
        seq indices[] = {s0, s1, s2, s3};
        //FIXME: check if this->s has same dimensions as numdims
        AF_THROW(af_weak_copy(&out, this->get()));
        return array(out, indices);
    }

    array array::row(int index) const
    {
        seq idx(index, index, 1);
        return this->operator()(idx, span, span, span);
    }

    array array::col(int index) const
    {
        seq idx(index, index, 1);
        return this->operator()(span, idx, span, span);
    }

    array array::slice(int index) const
    {
        seq idx(index, index, 1);
        return this->operator()(span, span, idx, span);
    }

    array array::rows(int first, int last) const
    {
        seq idx(first, last, 1);
        return this->operator()(idx, span, span, span);
    }

    array array::cols(int first, int last) const
    {
        seq idx(first, last, 1);
        return this->operator()(span, idx, span, span);
    }

    array array::slices(int first, int last) const
    {
        seq idx(first, last, 1);
        return this->operator()(span, span, idx, span);
    }

    array array::as(af_dtype type) const
    {
        af_array out;
        AF_THROW(af_cast(&out, this->get(), type));
        return array(out);
    }

    array::array(const array& in) : arr(0), isRef(false)
    {
        AF_THROW(af_weak_copy(&arr, in.get()));
    }

    // Transpose and Conjugate Transpose
    array array::T() const
    {
        return transpose(*this);
    }

    array array::H() const
    {
        return transpose(*this, true);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Operator =
    ///////////////////////////////////////////////////////////////////////////
    array& array::operator=(const array &other)
    {
        if (isRef) {

            af_seq afs[4];
            getSeq(afs);
            AF_THROW(af_assign(arr, numdims(), afs, other.get()));
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
        array cst = constant(value, this->dims(), this->type());
        return operator=(cst);
    }

    array& array::operator=(const cdouble &value)
    {
        array cst = constant(value, this->dims());
        return operator=(cst);
    }

    array& array::operator=(const cfloat &value)
    {
        array cst = constant(value, this->dims());
        return operator=(cst);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Operator +=, -=, *=, /=
    ///////////////////////////////////////////////////////////////////////////
#define INSTANTIATE(op, op1, func)                                  \
    array& array::operator op(const array &other)                   \
    {                                                               \
        bool this_ref = isRef;                                      \
        if (this_ref) {                                             \
            af_array tmp_arr;                                       \
            unsigned ndims = numdims();                             \
            AF_THROW(af_weak_copy(&tmp_arr, this->arr));            \
            array tmp = *this op1 other;                            \
            af_seq afs[4];                                          \
            getSeq(afs);                                            \
            AF_THROW(af_assign(tmp_arr, ndims, afs, tmp.get()));    \
            AF_THROW(af_destroy_array(this->arr));                  \
            this->arr = tmp_arr;                                    \
        } else {                                                    \
            *this = *this op1 other;                                \
        }                                                           \
        return *this;                                               \
    }                                                               \
    array& array::operator op(const double &value)                  \
    {                                                               \
        array cst = constant(value, this->dims(), this->type());    \
        return operator op(cst);                                    \
    }                                                               \
    array& array::operator op(const cdouble &value)                 \
    {                                                               \
        array cst = constant(value, this->dims());                  \
        return operator op(cst);                                    \
    }                                                               \
    array& array::operator op(const cfloat &value)                  \
    {                                                               \
        array cst = constant(value, this->dims());                  \
        return operator op(cst);                                    \
    }                                                               \

    INSTANTIATE(+=, +, af_add)
    INSTANTIATE(-=, -, af_sub)
    INSTANTIATE(*=, *, af_mul)
    INSTANTIATE(/=, /, af_div)

#undef INSTANTIATE

    ///////////////////////////////////////////////////////////////////////////
    // Operator +, -, *, /
    ///////////////////////////////////////////////////////////////////////////
#define INSTANTIATE(op, func)                                       \
    array array::operator op(const array &other) const              \
    {                                                               \
        af_array out;                                               \
        AF_THROW(func(&out, this->get(), other.get()));             \
        return array(out);                                          \
    }                                                               \
    array array::operator op(const double &value) const             \
    {                                                               \
        af_array out;                                               \
        array cst = constant(value, this->dims(), this->type());    \
        AF_THROW(func(&out, this->get(), cst.get()));               \
        return array(out);                                          \
    }                                                               \
    array array::operator op(const cdouble &value) const            \
    {                                                               \
        af_array out;                                               \
        array cst = constant(value, this->dims());                  \
        AF_THROW(func(&out, this->get(), cst.get()));               \
        return array(out);                                          \
    }                                                               \
    array array::operator op(const cfloat &value) const             \
    {                                                               \
        af_array out;                                               \
        array cst = constant(value, this->dims());                  \
        AF_THROW(func(&out, this->get(), cst.get()));               \
        return array(out);                                          \
    }                                                               \
    array operator op(const double &value, const array &other)      \
    {                                                               \
        af_array out;                                               \
        array cst = constant(value, other.dims(), other.type());    \
        AF_THROW(func(&out, cst.get(), other.get()));               \
        return array(out);                                          \
    }                                                               \
    array operator op(const cdouble &value, const array& other)     \
    {                                                               \
        af_array out;                                               \
        array cst = constant(value, other.dims());                  \
        AF_THROW(func(&out, cst.get(), other.get()));               \
        return array(out);                                          \
    }                                                               \
    array operator op(const cfloat &value, const array& other)      \
    {                                                               \
        af_array out;                                               \
        array cst = constant(value, other.dims());                  \
        AF_THROW(func(&out, cst.get(), other.get()));               \
        return array(out);                                          \
    }                                                               \

    INSTANTIATE(+, af_add)
    INSTANTIATE(-, af_sub)
    INSTANTIATE(*, af_mul)
    INSTANTIATE(/, af_div)

#undef INSTANTIATE

    ///////////////////////////////////////////////////////////////////////////
    // Operator ==, !=, < <=, >, >=
    ///////////////////////////////////////////////////////////////////////////
#define INSTANTIATE(op, func)                                       \
    array array::operator op(const array &other) const              \
    {                                                               \
    af_array out;                                                   \
    AF_THROW(func(&out, this->get(), other.get()));                 \
    return array(out);                                              \
}                                                                   \
    array array::operator op(const bool &value) const               \
    {                                                               \
    af_array out;                                                   \
    array cst = constant(value, this->dims(), this->type());        \
    AF_THROW(func(&out, this->get(), cst.get()));                   \
    return array(out);                                              \
}                                                                   \
    array array::operator op(const int &value) const                \
    {                                                               \
    af_array out;                                                   \
    array cst = constant(value, this->dims(), this->type());        \
    AF_THROW(func(&out, this->get(), cst.get()));                   \
    return array(out);                                              \
    }                                                               \
    array array::operator op(const double &value) const             \
    {                                                               \
        af_array out;                                               \
        array cst = constant(value, this->dims(), this->type());    \
        AF_THROW(func(&out, this->get(), cst.get()));               \
        return array(out);                                          \
    }                                                               \
    array array::operator op(const cdouble &value) const            \
    {                                                               \
        af_array out;                                               \
        array cst = constant(value, this->dims());                  \
        AF_THROW(func(&out, this->get(), cst.get()));               \
        return array(out);                                          \
    }                                                               \
    array array::operator op(const cfloat &value) const             \
    {                                                               \
        af_array out;                                               \
        array cst = constant(value, this->dims());                  \
        AF_THROW(func(&out, this->get(), cst.get()));               \
        return array(out);                                          \
    }                                                               \
    array operator op(const bool &value, const array &other)        \
    {                                                               \
        af_array out;                                               \
        array cst = constant(value, other.dims(), other.type());    \
        AF_THROW(func(&out, cst.get(), other.get()));               \
        return array(out);                                          \
    }                                                               \
    array operator op(const int &value, const array &other)         \
    {                                                               \
        af_array out;                                               \
        array cst = constant(value, other.dims(), other.type());    \
        AF_THROW(func(&out, cst.get(), other.get()));               \
        return array(out);                                          \
    }                                                               \
    array operator op(const double &value, const array &other)      \
    {                                                               \
        af_array out;                                               \
        array cst = constant(value, other.dims(), other.type());    \
        AF_THROW(func(&out, cst.get(), other.get()));               \
        return array(out);                                          \
    }                                                               \
    array operator op(const cdouble &value, const array& other)     \
    {                                                               \
        af_array out;                                               \
        array cst = constant(value, other.dims());                  \
        AF_THROW(func(&out, cst.get(), other.get()));               \
        return array(out);                                          \
    }                                                               \
    array operator op(const cfloat &value, const array& other)      \
    {                                                               \
        af_array out;                                               \
        array cst = constant(value, other.dims());                  \
        AF_THROW(func(&out, cst.get(), other.get()));               \
        return array(out);                                          \
    }                                                               \

        INSTANTIATE(==, af_eq)
        INSTANTIATE(!=, af_neq)
        INSTANTIATE(< , af_lt)
        INSTANTIATE(<=, af_le)
        INSTANTIATE(> , af_gt)
        INSTANTIATE(>=, af_ge)
        INSTANTIATE(&&, af_and)
        INSTANTIATE(||, af_or)
        INSTANTIATE(%, af_mod)

    array array::operator-() const
    {
        af_array out;
        array cst = constant(0, this->dims(), this->type());
        AF_THROW(af_sub(&out, cst.get(), this->get()));
        return array(out);
    }

    array array::operator!() const
    {
        af_array out;
        array cst = constant(0, this->dims(), this->type());
        AF_THROW(af_eq(&out, cst.get(), this->get()));
        return array(out);
    }

#undef INSTANTIATE

    void array::eval() const
    {
        AF_THROW(af_eval(get()));
    }

#define INSTANTIATE(T)                                      \
    template<> AFAPI T *array::host() const                 \
    {                                                       \
        if (type() != (af_dtype)dtype_traits<T>::af_type) { \
            AF_THROW(AF_ERR_INVALID_TYPE);                  \
        }                                                   \
                                                            \
        T *res = new T[elements()];                         \
        AF_THROW(af_get_data_ptr((void *)res, arr));        \
                                                            \
        return res;                                         \
    }                                                       \
    template<> AFAPI T array::scalar() const                \
    {                                                       \
        T *h_ptr = host<T>();                               \
        T scalar = h_ptr[0];                                \
        delete[] h_ptr;                                     \
        return scalar;                                      \
    }                                                       \
    template<> AFAPI T* array::device() const               \
    {                                                       \
        void *ptr = NULL;                                   \
        AF_THROW(af_get_device_ptr(&ptr, get(), true));     \
        return (T *)ptr;                                    \
    }                                                       \

    INSTANTIATE(cdouble)
    INSTANTIATE(cfloat)
    INSTANTIATE(double)
    INSTANTIATE(float)
    INSTANTIATE(unsigned)
    INSTANTIATE(int)
    INSTANTIATE(unsigned char)
    INSTANTIATE(char)
}
