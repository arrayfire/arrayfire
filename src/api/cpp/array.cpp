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
#include <af/gfor.h>
#include "error.hpp"

namespace af
{

    static int gforDim(std::vector<seq> &s)
    {
        for (int i = 0; i < 4; i++) {
            if (s[i].m_gfor) return i;
        }
        return -1;
    }

    static af_array gforReorder(const af_array in, unsigned dim)
    {
        unsigned order[4] = {0, 1, 2, dim};
        order[dim] = 3;
        af_array out;
        AF_THROW(af_reorder(&out, in, order[0], order[1], order[2], order[3]));
        return out;
    }

    static af::dim4 seqToDims(af_seq *afs, af::dim4 parentDims)
    {
        std::vector<af_seq> av(afs, afs + 4);
        return toDims(av, parentDims);
    }

    static unsigned size_of(af::dtype type)
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

    static unsigned numDims(const af_array arr)
    {
        unsigned nd;
        AF_THROW(af_get_numdims(&nd, arr));
        return nd;
    }

    static dim4 getDims(const af_array arr)
    {
        dim_type d0, d1, d2, d3;
        AF_THROW(af_get_dims(&d0, &d1, &d2, &d3, arr));
        return dim4(d0, d1, d2, d3);
    }

    array::array(const af_array handle): arr(handle), parent(NULL), isRef(false), s(4)
    {
    }

    static void initEmptyArray(af_array *arr, af::dtype ty,
                               dim_type d0, dim_type d1=1, dim_type d2=1, dim_type d3=1)
    {
        dim_type my_dims[] = {d0, d1, d2, d3};
        AF_THROW(af_create_handle(arr, 4, my_dims, ty));
    }

    template<typename T>
    static void initDataArray(af_array *arr, const T *ptr, af_source_t src,
                              dim_type d0, dim_type d1=1, dim_type d2=1, dim_type d3=1)
    {
        af::dtype ty = (af::dtype)dtype_traits<T>::af_type;
        dim_type my_dims[] = {d0, d1, d2, d3};
        switch (src) {
        case afHost:   AF_THROW(af_create_array(arr, (const void * const)ptr, 4, my_dims, ty)); break;
        case afDevice: AF_THROW(af_device_array(arr, (const void *      )ptr, 4, my_dims, ty)); break;
        default: AF_THROW_MSG("Can not create array from the requested source pointer",
                              AF_ERR_INVALID_ARG);
        }
    }

    array::array() : arr(0),  parent(NULL), isRef(false), s(4)
    {
        initEmptyArray(&arr, f32, 0, 0, 0, 0);
    }
    array::array(const dim4 &dims, af::dtype ty) : arr(0), parent(NULL), isRef(false), s(4)
    {
        initEmptyArray(&arr, ty, dims[0], dims[1], dims[2], dims[3]);
    }

    array::array(dim_type d0, af::dtype ty) : arr(0), parent(NULL), isRef(false), s(4)
    {
        initEmptyArray(&arr, ty, d0);
    }

    array::array(dim_type d0, dim_type d1, af::dtype ty) : arr(0), parent(NULL), isRef(false), s(4)
    {
        initEmptyArray(&arr, ty, d0, d1);
    }

    array::array(dim_type d0, dim_type d1, dim_type d2, af::dtype ty) :
        arr(0), parent(NULL), isRef(false), s(4)
    {
        initEmptyArray(&arr, ty, d0, d1, d2);
    }

    array::array(dim_type d0, dim_type d1, dim_type d2, dim_type d3, af::dtype ty) :
        arr(0), parent(NULL), isRef(false), s(4)
    {
        initEmptyArray(&arr, ty, d0, d1, d2, d3);
    }

#define INSTANTIATE(T)                                                  \
    template<> AFAPI                                                    \
    array::array(const dim4 &dims, const T *ptr, af_source_t src, dim_type ngfor) \
        : arr(0), parent(NULL), isRef(false), s(4)                      \
    {                                                                   \
        initDataArray<T>(&arr, ptr, src, dims[0], dims[1], dims[2], dims[3]); \
    }                                                                   \
    template<> AFAPI                                                    \
    array::array(dim_type d0, const T *ptr, af_source_t src, dim_type ngfor) \
        : arr(0), parent(NULL), isRef(false), s(4)                      \
    {                                                                   \
        initDataArray<T>(&arr, ptr, src, d0);                           \
    }                                                                   \
    template<> AFAPI                                                    \
    array::array(dim_type d0, dim_type d1, const T *ptr, af_source_t src, \
                 dim_type ngfor) : arr(0), parent(NULL), isRef(false), s(4) \
    {                                                                   \
        initDataArray<T>(&arr, ptr, src, d0, d1);                       \
    }                                                                   \
    template<> AFAPI                                                    \
    array::array(dim_type d0, dim_type d1, dim_type d2, const T *ptr,   \
                 af_source_t src, dim_type ngfor) :                     \
        arr(0), parent(NULL), isRef(false), s(4)                        \
    {                                                                   \
        initDataArray<T>(&arr, ptr, src, d0, d1, d2);                   \
    }                                                                   \
    template<> AFAPI                                                    \
    array::array(dim_type d0, dim_type d1, dim_type d2, dim_type d3, const T *ptr, \
                 af_source_t src, dim_type ngfor) :                     \
        arr(0), parent(NULL), isRef(false), s(4)                        \
                                                                        \
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
    INSTANTIATE(intl)
    INSTANTIATE(uintl)

#undef INSTANTIATE

    array::~array()
    {
        af_array tmp = get();
        if (tmp != 0) AF_THROW(af_destroy_array(tmp));
    }

    af::dtype array::type() const
    {
        af::dtype my_type;
        AF_THROW(af_get_type(&my_type, arr));
        return my_type;
    }

    dim_type array::elements() const
    {
        dim_type elems;
        AF_THROW(af_get_elements(&elems, get()));
        return elems;
    }

    void array::host(void *data) const
    {
        AF_THROW(af_get_data_ptr(data, get()));
    }

    af_array array::get()
    {
        if (!isRef)
            return arr;
        af_array temp = 0;
        af_seq afs[4];
        getSeq(afs);
        af_err err = af_index(&temp, arr, 4, afs);
        AF_THROW(af_destroy_array(arr));

        int dim = gforDim(this->s);
        if (temp && dim >= 0) {
            arr = gforReorder(temp, dim);
            AF_THROW(af_destroy_array(temp));
        } else {
            arr = temp;
        }

        isRef = false;
        AF_THROW(err);
        return arr;
    }

    af_array array::get() const
    {
        return ((array *)(this))->get();
    }

    // Helper functions
    dim4 array::dims() const
    {
        return getDims(get());
    }

    dim_type array::dims(unsigned dim) const
    {
        return dims()[dim];
    }

    unsigned array::numdims() const
    {
        return numDims(get());
    }

    size_t array::bytes() const
    {
        dim_type nElements;
        AF_THROW(af_get_elements(&nElements, get()));
        return nElements * size_of(type());
    }

    array array::copy() const
    {
        af_array other = 0;
        AF_THROW(af_copy_array(&other, get()));
        return array(other);
    }

#undef INSTANTIATE
#define INSTANTIATE(fn)                         \
    bool array::is##fn() const                  \
    {                                           \
        bool ret = false;                       \
        AF_THROW(af_is_##fn(&ret, get()));      \
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

    array::array(af_array in, const array *par, seq *seqs) : arr(in), parent(par), isRef(true), s(4)
    {
        for(int i=0; i<4; ++i) s[i] = seqs[i];
    }

    void array::getSeq(af_seq* afs) const
    {
        afs[0] = this->s[0].s;
        afs[1] = this->s[1].s;
        afs[2] = this->s[2].s;
        afs[3] = this->s[3].s;
    }

    array array::operator()(const array& idx) const
    {
        eval();
        af_array out = 0;
        AF_THROW(af_array_index(&out, this->get(), idx.get(), 0));
        return array(out);
    }

    array array::operator()(const seq &s0) const
    {
        eval();
        af_array out = 0;
        seq indices[] = {s0, span, span, span};
        //FIXME: check if this->s has same dimensions as numdims
        AF_THROW(af_weak_copy(&out, this->get()));
        return array(out, this, indices);
    }

    array array::operator()(const seq &s0, const seq &s1) const
    {
        eval();
        af_array out = 0;
        seq indices[] = {s0, s1, span, span};
        //FIXME: check if this->s has same dimensions as numdims
        AF_THROW(af_weak_copy(&out, this->get()));
        return array(out, this, indices);
    }

    array array::operator()(const seq &s0, const seq &s1, const seq &s3) const
    {
        eval();
        af_array out = 0;
        seq indices[] = {s0, s1, s3, span};
        //FIXME: check if this->s has same dimensions as numdims
        AF_THROW(af_weak_copy(&out, this->get()));
        return array(out, this, indices);
    }

    array array::operator()(const seq &s0, const seq &s1, const seq &s2, const seq &s3) const
    {
        eval();
        af_array out = 0;
        seq indices[] = {s0, s1, s2, s3};
        //FIXME: check if this->s has same dimensions as numdims
        AF_THROW(af_weak_copy(&out, this->get()));
        return array(out, this, indices);
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

    array array::as(af::dtype type) const
    {
        af_array out;
        AF_THROW(af_cast(&out, this->get(), type));
        return array(out);
    }

    array::array(const array& in) : arr(0), parent(NULL), isRef(false), s(4)
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

    void array::set(af_array tmp)
    {
        AF_THROW(af_destroy_array(arr));
        arr = tmp;
    }

    void array::set(af_array tmp) const
    {
        ((array *)(this))->set(tmp);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Operator =
    ///////////////////////////////////////////////////////////////////////////
    array& array::operator=(const array &other)
    {
        if (isRef) {

            af_seq afs[4];
            getSeq(afs);
            unsigned nd = numDims(arr);
            int dim = gforDim(this->s);
            af_array other_arr = other.get();

            // HACK: This is a quick check to see if other has been reordered inside gfor
            // TODO: Figure out if this breaks and implement a cleaner method
            bool is_reordered = (getDims(arr) != other.dims());

            other_arr = (dim == -1 || !is_reordered) ? other_arr : gforReorder(other_arr, dim);

            af_array tmp;
            AF_THROW(af_assign(&tmp, arr, nd, afs, other_arr));
            parent->set(tmp);

            if (dim >= 0 && is_reordered) AF_THROW(af_destroy_array(other_arr));

            isRef = false;

        } else {

            if (this->get() == other.get()) {
                return *this;
            }
            if(this->arr != 0) {
                AF_THROW(af_destroy_array(this->arr));
            }

            af_array temp = 0;
            AF_THROW(af_weak_copy(&temp, other.get()));
            this->arr = temp;
        }
        return *this;
    }

#define SELF_OP(op, op1)                                                \
    array& array::operator op(const array &other)                       \
    {                                                                   \
        bool this_ref = isRef;                                          \
        if (this_ref) {                                                 \
            af_array lhs;                                               \
            int dim = gforDim(this->s);                                 \
            AF_THROW(af_weak_copy(&lhs, this->arr));                    \
            unsigned ndims = numDims(lhs);                              \
            /* FIXME: Unify with other af_assign */                     \
            array tmp = *this op1 other;                                \
            af_seq afs[4];                                              \
            getSeq(afs);                                                \
            af_array tmp_arr = tmp.get();                               \
            af_array out = 0;                                           \
            tmp_arr = (dim == -1) ? tmp_arr : gforReorder(tmp_arr, dim); \
            AF_THROW(af_assign(&out, lhs, ndims, afs, tmp_arr));        \
            AF_THROW(af_destroy_array(this->arr));                      \
            if (dim >= 0) AF_THROW(af_destroy_array(tmp_arr));          \
            this->arr = lhs;                                            \
            parent->set(out);                                           \
        } else {                                                        \
            *this = *this op1 other;                                    \
        }                                                               \
        return *this;                                                   \
    }                                                                   \

    SELF_OP(+=, +)
    SELF_OP(-=, -)
    SELF_OP(*=, *)
    SELF_OP(/=, /)

#undef SELF_OP

#define ASSIGN_TYPE(TY, OP)                                 \
    array& array::operator OP(const TY &value)              \
    {                                                       \
        af_seq afs[4];                                      \
        getSeq(afs);                                        \
        af::dim4 dims = isRef ?                             \
            seqToDims(afs, getDims(arr)) : this->dims();    \
        array cst = constant(value, dims, this->type());    \
        return operator OP(cst);                            \
    }                                                       \

#define ASSIGN_OP(OP)                           \
    ASSIGN_TYPE(double, OP)                     \
    ASSIGN_TYPE(float, OP)                      \
    ASSIGN_TYPE(cdouble, OP)                    \
    ASSIGN_TYPE(cfloat, OP)                     \
    ASSIGN_TYPE(int, OP)                        \
    ASSIGN_TYPE(unsigned, OP)                   \
    ASSIGN_TYPE(long, OP)                       \
    ASSIGN_TYPE(unsigned long, OP)              \
    ASSIGN_TYPE(long long, OP)                  \
    ASSIGN_TYPE(unsigned long long, OP)         \
    ASSIGN_TYPE(char, OP)                       \
    ASSIGN_TYPE(unsigned char, OP)              \
    ASSIGN_TYPE(bool, OP)                       \

    ASSIGN_OP(= )
    ASSIGN_OP(+=)
    ASSIGN_OP(-=)
    ASSIGN_OP(*=)
    ASSIGN_OP(/=)

#undef ASSIGN_OP
#undef ASSIGN_TYPE

#define BINARY_TYPE(TY, OP, func)                                   \
    array array::operator OP(const TY &value) const                 \
    {                                                               \
        af_array lhs = this->get();                                 \
        af_array out;                                               \
        array cst = constant(value, this->dims(), this->type());    \
        AF_THROW(func(&out, lhs, cst.get(), gforGet()));            \
        return array(out);                                          \
    }                                                               \
    array operator OP(const TY &value, const array &other)          \
    {                                                               \
        af_array rhs = other.get();                                 \
        af_array out;                                               \
        array cst = constant(value, other.dims(), other.type());    \
        AF_THROW(func(&out, cst.get(), rhs, gforGet()));            \
        return array(out);                                          \
    }                                                               \

#define BINARY_OP(OP, func)                                 \
    array array::operator OP(const array &other) const      \
    {                                                       \
        af_array lhs = this->get();                         \
        af_array out;                                       \
        AF_THROW(func(&out, lhs, other.get(), gforGet()));  \
        return array(out);                                  \
    }                                                       \
    BINARY_TYPE(double             , OP, func)              \
    BINARY_TYPE(float              , OP, func)              \
    BINARY_TYPE(cdouble            , OP, func)              \
    BINARY_TYPE(cfloat             , OP, func)              \
    BINARY_TYPE(int                , OP, func)              \
    BINARY_TYPE(unsigned           , OP, func)              \
    BINARY_TYPE(long               , OP, func)              \
    BINARY_TYPE(unsigned long      , OP, func)              \
    BINARY_TYPE(long long          , OP, func)              \
    BINARY_TYPE(unsigned long long , OP, func)              \
    BINARY_TYPE(char               , OP, func)              \
    BINARY_TYPE(unsigned char      , OP, func)              \
    BINARY_TYPE(bool               , OP, func)              \

    BINARY_OP(+, af_add)
    BINARY_OP(-, af_sub)
    BINARY_OP(*, af_mul)
    BINARY_OP(/, af_div)
    BINARY_OP(==, af_eq)
    BINARY_OP(!=, af_neq)
    BINARY_OP(< , af_lt)
    BINARY_OP(<=, af_le)
    BINARY_OP(> , af_gt)
    BINARY_OP(>=, af_ge)
    BINARY_OP(&&, af_and)
    BINARY_OP(||, af_or)
    BINARY_OP(%, af_rem)
    BINARY_OP(&, af_bitand)
    BINARY_OP(|, af_bitor)
    BINARY_OP(^, af_bitxor)
    BINARY_OP(<<, af_bitshiftl)
    BINARY_OP(>>, af_bitshiftr)

#undef BINARY_TYPE
#undef BINARY_OP


    array array::operator-() const
    {
        af_array lhs = this->get();
        af_array out;
        array cst = constant(0, this->dims(), this->type());
        AF_THROW(af_sub(&out, cst.get(), lhs, gforGet()));
        return array(out);
    }

    array array::operator!() const
    {
        af_array lhs = this->get();
        af_array out;
        array cst = constant(0, this->dims(), this->type());
        AF_THROW(af_eq(&out, cst.get(), lhs, gforGet()));
        return array(out);
    }

    void array::eval() const
    {
        AF_THROW(af_eval(get()));
    }

#define INSTANTIATE(T)                                              \
    template<> AFAPI T *array::host() const                         \
    {                                                               \
        if (type() != (af::dtype)dtype_traits<T>::af_type) {        \
            AF_THROW_MSG("Requested type does'nt match with array", \
                         AF_ERR_INVALID_TYPE);                      \
        }                                                           \
                                                                    \
        T *res = new T[elements()];                                 \
        AF_THROW(af_get_data_ptr((void *)res, get()));              \
                                                                    \
        return res;                                                 \
    }                                                               \
    template<> AFAPI T array::scalar() const                        \
    {                                                               \
        T *h_ptr = host<T>();                                       \
        T scalar = h_ptr[0];                                        \
        delete[] h_ptr;                                             \
        return scalar;                                              \
    }                                                               \
    template<> AFAPI T* array::device() const                       \
    {                                                               \
        void *ptr = NULL;                                           \
        AF_THROW(af_get_device_ptr(&ptr, get(), true));             \
        return (T *)ptr;                                            \
    }                                                               \

    INSTANTIATE(cdouble)
    INSTANTIATE(cfloat)
    INSTANTIATE(double)
    INSTANTIATE(float)
    INSTANTIATE(unsigned)
    INSTANTIATE(int)
    INSTANTIATE(unsigned char)
    INSTANTIATE(char)
    INSTANTIATE(intl)
    INSTANTIATE(uintl)

}
