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
#include <af/algorithm.h>
#include <af/internal.h>
#include "error.hpp"

#include <stdexcept>
#include <memory>

namespace af
{
    static int gforDim(af_index_t *indices)
    {
        for (int i = 0; i < AF_MAX_DIMS; i++) {
            if (indices[i].isBatch) return i;
        }
        return -1;
    }

    static af_array gforReorder(const af_array in, unsigned dim)
    {
        // This is here to stop gcc from complaining
        if (dim > 3) AF_THROW_ERR("GFor: Dimension is invalid", AF_ERR_SIZE);
        unsigned order[AF_MAX_DIMS] = {0, 1, 2, dim};
        order[dim] = 3;
        af_array out;
        AF_THROW(af_reorder(&out, in, order[0], order[1], order[2], order[3]));
        return out;
    }

    static af::dim4 seqToDims(af_index_t *indices, af::dim4 parentDims, bool reorder = true)
    {
        try {
            af::dim4 odims(1);
            for (int i = 0; i < AF_MAX_DIMS; i++) {
                if (indices[i].isSeq) {
                    odims[i] = calcDim(indices[i].idx.seq, parentDims[i]);
                } else {
                    dim_t elems = 0;
                    AF_THROW(af_get_elements(&elems, indices[i].idx.arr));
                    odims[i] = elems;
                }
            }

            // Change the dimensions if inside GFOR
            if (reorder) {
                for (int i = 0; i < AF_MAX_DIMS; i++) {
                    if (indices[i].isBatch) {
                        int tmp = odims[i];
                        odims[i] = odims[3];
                        odims[3] = tmp;
                        break;
                    }
                }
            }
            return odims;
        } catch(std::logic_error &err) {
            AF_THROW_ERR(err.what(), AF_ERR_SIZE);
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
        dim_t d0, d1, d2, d3;
        AF_THROW(af_get_dims(&d0, &d1, &d2, &d3, arr));
        return dim4(d0, d1, d2, d3);
    }

    struct array::array_proxy::array_proxy_impl
    {
        array *       parent_;        //< The original array
        af_index_t    indices_[4];    //< Indexing array or seq objects
        bool          is_linear_;

        // if true the parent_ object will be deleted on distruction. This is
        // necessary only when calling indexing functions in array_proxy objects.
        bool          delete_on_destruction_;
        array_proxy_impl(array &parent, af_index_t *idx, bool linear)
            : parent_(&parent)
            , indices_()
            , is_linear_(linear)
            , delete_on_destruction_(false)
        {
            std::copy(idx, idx + AF_MAX_DIMS, indices_);
        }

        void delete_on_destruction(bool val) {
            delete_on_destruction_ = val;
        }

        ~array_proxy_impl() {
            if (delete_on_destruction_) delete parent_;
        }
      private:
        array_proxy_impl(const array_proxy_impl&);
        array_proxy_impl(const array_proxy_impl&&);
        array_proxy_impl operator=(const array_proxy_impl&);
        array_proxy_impl operator=(const array_proxy_impl&&);
    };

    array::array(const af_array handle): arr(handle)
    {
    }

    static void initEmptyArray(af_array *arr, af::dtype ty,
                               dim_t d0, dim_t d1=1, dim_t d2=1, dim_t d3=1)
    {
        dim_t my_dims[] = {d0, d1, d2, d3};
        AF_THROW(af_create_handle(arr, AF_MAX_DIMS, my_dims, ty));
    }

    template<typename T>
    static void initDataArray(af_array *arr, const T *ptr, af::source src,
                              dim_t d0, dim_t d1=1, dim_t d2=1, dim_t d3=1)
    {
        af::dtype ty = (af::dtype)dtype_traits<T>::af_type;
        dim_t my_dims[] = {d0, d1, d2, d3};
        switch (src) {
        case afHost:   AF_THROW(af_create_array(arr, (const void * const)ptr, AF_MAX_DIMS, my_dims, ty)); break;
        case afDevice: AF_THROW(af_device_array(arr, (const void *      )ptr, AF_MAX_DIMS, my_dims, ty)); break;
        default: AF_THROW_ERR("Can not create array from the requested source pointer",
                              AF_ERR_ARG);
        }
    }

    array::array() : arr(0)
    {
        initEmptyArray(&arr, f32, 0, 1, 1, 1);
    }

    array::array(const dim4 &dims, af::dtype ty) : arr(0)
    {
        initEmptyArray(&arr, ty, dims[0], dims[1], dims[2], dims[3]);
    }

    array::array(dim_t d0, af::dtype ty) : arr(0)
    {
        initEmptyArray(&arr, ty, d0);
    }

    array::array(dim_t d0, dim_t d1, af::dtype ty) : arr(0)
    {
        initEmptyArray(&arr, ty, d0, d1);
    }

    array::array(dim_t d0, dim_t d1, dim_t d2, af::dtype ty) :
        arr(0)
    {
        initEmptyArray(&arr, ty, d0, d1, d2);
    }

    array::array(dim_t d0, dim_t d1, dim_t d2, dim_t d3, af::dtype ty) :
        arr(0)
    {
        initEmptyArray(&arr, ty, d0, d1, d2, d3);
    }

#define INSTANTIATE(T)                                                  \
    template<> AFAPI                                                    \
    array::array(const dim4 &dims, const T *ptr, af::source src)        \
        : arr(0)                                                        \
    {                                                                   \
        initDataArray<T>(&arr, ptr, src, dims[0], dims[1], dims[2],     \
                dims[3]);                                               \
    }                                                                   \
    template<> AFAPI                                                    \
    array::array(dim_t d0, const T *ptr, af::source src)                \
        : arr(0)                                                        \
    {                                                                   \
        initDataArray<T>(&arr, ptr, src, d0);                           \
    }                                                                   \
    template<> AFAPI                                                    \
    array::array(dim_t d0, dim_t d1, const T *ptr, af::source src)      \
        : arr(0)                                                        \
    {                                                                   \
        initDataArray<T>(&arr, ptr, src, d0, d1);                       \
    }                                                                   \
    template<> AFAPI                                                    \
    array::array(dim_t d0, dim_t d1, dim_t d2, const T *ptr,            \
                 af::source src)                                        \
        : arr(0)                                                        \
    {                                                                   \
        initDataArray<T>(&arr, ptr, src, d0, d1, d2);                   \
    }                                                                   \
    template<> AFAPI                                                    \
    array::array(dim_t d0, dim_t d1, dim_t d2, dim_t d3, const T *ptr,  \
                 af::source src) :                                      \
        arr(0)                                                          \
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
    INSTANTIATE(short)
    INSTANTIATE(unsigned short)

#undef INSTANTIATE

    array::~array()
    {
        af_array tmp = get();
        // THOU SHALL NOT THROW IN DESTRUCTORS
        af_release_array(tmp);
    }

    af::dtype array::type() const
    {
        af::dtype my_type;
        AF_THROW(af_get_type(&my_type, arr));
        return my_type;
    }

    dim_t array::elements() const
    {
        dim_t elems;
        AF_THROW(af_get_elements(&elems, get()));
        return elems;
    }

    void array::host(void *data) const
    {
        AF_THROW(af_get_data_ptr(data, get()));
    }

    af_array array::get()
    {
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

    dim_t array::dims(unsigned dim) const
    {
        return dims()[dim];
    }

    unsigned array::numdims() const
    {
        return numDims(get());
    }

    size_t array::bytes() const
    {
        dim_t nElements;
        AF_THROW(af_get_elements(&nElements, get()));
        return nElements * getSizeOf(type());
    }

    size_t array::allocated() const
    {
        size_t result = 0;
        AF_THROW(af_get_allocated_bytes(&result, get()));
        return result;
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
    INSTANTIATE(bool)
    INSTANTIATE(sparse)

#undef INSTANTIATE

    static array::array_proxy gen_indexing(const array &ref, const index &s0, const index &s1, const index &s2, const index &s3, bool linear = false)
    {
        ref.eval();
        af_index_t inds[AF_MAX_DIMS];
        inds[0] = s0.get();
        inds[1] = s1.get();
        inds[2] = s2.get();
        inds[3] = s3.get();

        return array::array_proxy(const_cast<array&>(ref), inds, linear);
    }

    array::array_proxy array::operator()(const index &s0)
    {
        return const_cast<const array*>(this)->operator()(s0);
    }

    array::array_proxy array::operator()(const index &s0, const index &s1, const index &s2, const index &s3)
    {
        return const_cast<const array*>(this)->operator()(s0, s1, s2, s3);
    }

    const array::array_proxy array::operator()(const index &s0) const
    {
        index z = index(0);
        if(isvector()){
            switch(numDims(this->arr)) {
                case 1: return gen_indexing(*this, s0, z, z, z);
                case 2: return gen_indexing(*this, z, s0, z, z);
                case 3: return gen_indexing(*this, z, z, s0, z);
                case 4: return gen_indexing(*this, z, z, z, s0);
                default: AF_THROW_ERR("ndims for Array is invalid", AF_ERR_SIZE);
            }
        }
        else {
            return gen_indexing(*this, s0, z, z, z, true);
        }
    }

    const array::array_proxy array::operator()(const index &s0, const index &s1, const index &s2, const index &s3) const
    {
        return gen_indexing(*this, s0, s1, s2, s3);
    }

    const array::array_proxy array::row(int index) const
    {
        return this->operator()(index, span, span, span);
    }

    array::array_proxy array::row(int index)
    {
        return const_cast<const array*>(this)->row(index);
    }

    const array::array_proxy array::col(int index) const
    {
        return this->operator()(span, index, span, span);
    }

    array::array_proxy array::col(int index)
    {
        return const_cast<const array*>(this)->col(index);
    }

    const array::array_proxy array::slice(int index) const
    {
        return this->operator()(span, span, index, span);
    }

    array::array_proxy array::slice(int index)
    {
        return const_cast<const array*>(this)->slice(index);
    }

    const array::array_proxy array::rows(int first, int last) const
    {
        seq idx(first, last, 1);
        return this->operator()(idx, span, span, span);
    }

    array::array_proxy array::rows(int first, int last)
    {
        return const_cast<const array*>(this)->rows(first, last);
    }

    const array::array_proxy array::cols(int first, int last) const
    {
        seq idx(first, last, 1);
        return this->operator()(span, idx, span, span);
    }

    array::array_proxy array::cols(int first, int last)
    {
        return const_cast<const array*>(this)->cols(first, last);
    }

    const array::array_proxy array::slices(int first, int last) const
    {
        seq idx(first, last, 1);
        return this->operator()(span, span, idx, span);
    }

    array::array_proxy array::slices(int first, int last)
    {
        return const_cast<const array*>(this)->slices(first, last);
    }

    const array array::as(af::dtype type) const
    {
        af_array out;
        AF_THROW(af_cast(&out, this->get(), type));
        return array(out);
    }

    array::array(const array& in) : arr(0)
    {
        AF_THROW(af_retain_array(&arr, in.get()));
    }

    array::array(const array& input, const dim4& dims) : arr(0)
    {
        AF_THROW(af_moddims(&arr, input.get(), AF_MAX_DIMS, dims.get()));
    }

    array::array(const array& input, const dim_t dim0, const dim_t dim1, const dim_t dim2, const dim_t dim3)
        : arr(0)
    {
        dim_t dims[] = {dim0, dim1, dim2, dim3};
        AF_THROW(af_moddims(&arr, input.get(), AF_MAX_DIMS, dims));
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
        if (arr) AF_THROW(af_release_array(arr));
        arr = tmp;
    }

    // Assign values to an array
    array::array_proxy&
    af::array::array_proxy::operator=(const array &other)
    {
        unsigned nd = numDims(impl->parent_->get());
        const dim4 this_dims = getDims(impl->parent_->get());
        const dim4 other_dims = other.dims();
        int dim = gforDim(impl->indices_);
        af_array other_arr = other.get();

        bool batch_assign = false;
        bool is_reordered = false;
        if (dim >= 0) {
            //FIXME: Figure out a faster, cleaner way to do this
            dim4 out_dims = seqToDims(impl->indices_, this_dims, false);

            batch_assign = true;
            for (int i = 0; i < AF_MAX_DIMS; i++) {
                if (this->impl->indices_[i].isBatch) batch_assign &= (other_dims[i] == 1);
                else                          batch_assign &= (other_dims[i] == out_dims[i]);
            }

            if (batch_assign) {
                af_array out;
                AF_THROW(af_tile(&out, other_arr,
                                 out_dims[0] / other_dims[0],
                                 out_dims[1] / other_dims[1],
                                 out_dims[2] / other_dims[2],
                                 out_dims[3] / other_dims[3]));
                other_arr = out;

            } else if (out_dims != other_dims) {
                // HACK: This is a quick check to see if other has been reordered inside gfor
                // TODO: Figure out if this breaks and implement a cleaner method
                other_arr = gforReorder(other_arr, dim);
                is_reordered = true;
            }
        }

        af_array par_arr = 0;

        if (impl->is_linear_) {
            AF_THROW(af_flat(&par_arr, impl->parent_->get()));
            nd = 1;
        } else {
            par_arr = impl->parent_->get();
        }

        af_array tmp = 0;
        AF_THROW(af_assign_gen(&tmp, par_arr, nd, impl->indices_, other_arr));

        af_array res = 0;
        if (impl->is_linear_) {
            AF_THROW(af_moddims(&res, tmp, this_dims.ndims(), this_dims.get()));
            AF_THROW(af_release_array(par_arr));
            AF_THROW(af_release_array(tmp));
        } else {
            res = tmp;
        }

        impl->parent_->set(res);

        if (dim >= 0 && (is_reordered || batch_assign)) {
            if (other_arr) AF_THROW(af_release_array(other_arr));
        }
        return *this;
    }

    array::array_proxy&
    af::array::array_proxy::operator=(const array::array_proxy &other)
    {
        array out = other;
        return *this = out;
    }

    af::array::array_proxy::array_proxy(array& par, af_index_t *ssss, bool linear)
        : impl(new array_proxy_impl(par, ssss, linear))
    {
    }

    af::array::array_proxy::array_proxy(const array_proxy &other)
        : impl(new array_proxy_impl(*other.impl->parent_, other.impl->indices_, other.impl->is_linear_))
    {
    }

#if __cplusplus > 199711L
    af::array::array_proxy::array_proxy(array_proxy &&other) {
        impl = other.impl;
    }

    array::array_proxy&
    af::array::array_proxy::operator=(array_proxy &&other) {
        array out = other;
        return *this = out;
    }
#endif

    af::array::array_proxy::~array_proxy() {
        if(impl) delete impl;
    }

    array array::array_proxy::as(dtype type) const
    {
        array out = *this;
        return out.as(type);
    }

    dim_t array::array_proxy::dims(unsigned dim) const
    {
        array out = *this;
        return out.dims(dim);
    }

    void array::array_proxy::host(void *ptr) const
    {
        array out = *this;
        return out.host(ptr);
    }

#define MEM_FUNC(PREFIX, FUNC)                  \
    PREFIX array::array_proxy::FUNC() const     \
    {                                           \
        array out = *this;                      \
        return out.FUNC();                      \
    }

    MEM_FUNC(dim_t                  , elements)
    MEM_FUNC(array                  , T)
    MEM_FUNC(array                  , H)
    MEM_FUNC(dtype                  , type)
    MEM_FUNC(dim4                   , dims)
    MEM_FUNC(unsigned               , numdims)
    MEM_FUNC(size_t                 , bytes)
    MEM_FUNC(size_t                 , allocated)
    MEM_FUNC(array                  , copy)
    MEM_FUNC(bool                   , isempty)
    MEM_FUNC(bool                   , isscalar)
    MEM_FUNC(bool                   , isvector)
    MEM_FUNC(bool                   , isrow)
    MEM_FUNC(bool                   , iscolumn)
    MEM_FUNC(bool                   , iscomplex)
    MEM_FUNC(bool                   , isdouble)
    MEM_FUNC(bool                   , issingle)
    MEM_FUNC(bool                   , isrealfloating)
    MEM_FUNC(bool                   , isfloating)
    MEM_FUNC(bool                   , isinteger)
    MEM_FUNC(bool                   , isbool)
    MEM_FUNC(bool                   , issparse)
    MEM_FUNC(void                   , eval)
    MEM_FUNC(af_array               , get)
    //MEM_FUNC(void                   , unlock)
#undef MEM_FUNC


#define ASSIGN_TYPE(TY, OP)                                   \
    array::array_proxy&                                       \
    array::array_proxy::operator OP(const TY &value)          \
    {                                                         \
        dim4 pdims = getDims(impl->parent_->get());           \
        if (impl->is_linear_) pdims = dim4(pdims.elements()); \
        dim4 dims = seqToDims(impl->indices_, pdims );        \
        af::dtype ty = impl->parent_->type();                 \
        array cst = constant(value, dims, ty);                \
        return this->operator OP(cst);                        \
    }                                                         \

#define ASSIGN_OP(OP, op1)                      \
    ASSIGN_TYPE(double             , OP)        \
    ASSIGN_TYPE(float              , OP)        \
    ASSIGN_TYPE(cdouble            , OP)        \
    ASSIGN_TYPE(cfloat             , OP)        \
    ASSIGN_TYPE(int                , OP)        \
    ASSIGN_TYPE(unsigned           , OP)        \
    ASSIGN_TYPE(long               , OP)        \
    ASSIGN_TYPE(unsigned long      , OP)        \
    ASSIGN_TYPE(long long          , OP)        \
    ASSIGN_TYPE(unsigned long long , OP)        \
    ASSIGN_TYPE(char               , OP)        \
    ASSIGN_TYPE(unsigned char      , OP)        \
    ASSIGN_TYPE(bool               , OP)        \
    ASSIGN_TYPE(short              , OP)        \
    ASSIGN_TYPE(unsigned short     , OP)        \

    ASSIGN_OP(= , =)
    ASSIGN_OP(+=, +)
    ASSIGN_OP(-=, -)
    ASSIGN_OP(*=, *)
    ASSIGN_OP(/=, /)
#undef ASSIGN_OP

#undef ASSIGN_TYPE

#define SELF_OP(OP, op1)                                                          \
    array::array_proxy& array::array_proxy::operator OP(const array_proxy &other) \
    {                                                                             \
        *this = *this op1 other;                                                  \
        return *this;                                                             \
    }                                                                             \
    array::array_proxy& array::array_proxy::operator OP(const array &other)       \
    {                                                                             \
        *this = *this op1 other;                                                  \
        return *this;                                                             \
    }                                                                             \

    SELF_OP(+=, +)
    SELF_OP(-=, -)
    SELF_OP(*=, *)
    SELF_OP(/=, /)
#undef SELF_OP

    array::array_proxy::operator array() const
    {
        af_array tmp = 0;
        af_array arr = 0;

        if(impl->is_linear_)  {
            AF_THROW(af_flat(&arr, impl->parent_->get()));
        } else {
            arr = impl->parent_->get();
        }

        AF_THROW(af_index_gen(&tmp, arr, AF_MAX_DIMS, impl->indices_));
        if (impl->is_linear_)  {
            AF_THROW(af_release_array(arr));
        }

        return array(tmp);
    }

    array::array_proxy::operator array()
    {
        af_array tmp = 0;
        af_array arr = 0;

        if(impl->is_linear_)  {
            AF_THROW(af_flat(&arr, impl->parent_->get()));
        } else {
            arr = impl->parent_->get();
        }

        AF_THROW(af_index_gen(&tmp, arr, AF_MAX_DIMS, impl->indices_));
        if(impl->is_linear_)  {
            AF_THROW(af_release_array(arr));
        }

        int dim = gforDim(impl->indices_);
        if (tmp && dim >= 0) {
            arr = gforReorder(tmp, dim);
            if (tmp) AF_THROW(af_release_array(tmp));
        } else {
            arr = tmp;
        }

        return array(arr);
    }

#define MEM_INDEX(FUNC_SIG, USAGE)                    \
    array::array_proxy                                \
    array::array_proxy::FUNC_SIG                      \
    {                                                 \
        array* out = new array(*this);                \
        array::array_proxy proxy = out->USAGE;        \
        proxy.impl->delete_on_destruction(true);      \
        return proxy;                                 \
    }                                                 \
                                                      \
    const array::array_proxy                          \
    array::array_proxy::FUNC_SIG const                \
    {                                                 \
        const array* out = new array(*this);          \
        array::array_proxy proxy = out->USAGE;        \
        proxy.impl->delete_on_destruction(true);      \
        return proxy;                                 \
    }

    MEM_INDEX(row(int index)                , row(index));
    MEM_INDEX(rows(int first, int last)     , rows(first, last));
    MEM_INDEX(col(int index)                , col(index));
    MEM_INDEX(cols(int first, int last)     , cols(first, last));
    MEM_INDEX(slice(int index)              , slice(index));
    MEM_INDEX(slices(int first, int last)   , slices(first, last));

#undef MEM_INDEX

    ///////////////////////////////////////////////////////////////////////////
    // Operator =
    ///////////////////////////////////////////////////////////////////////////
    array& array::operator=(const array &other)
    {
        if (this->get() == other.get()) {
            return *this;
        }
        //TODO: Unsafe. loses data if af_weak_copy fails
        if(this->arr != 0) {
            AF_THROW(af_release_array(this->arr));
        }

        af_array temp = 0;
        AF_THROW(af_retain_array(&temp, other.get()));
        this->arr = temp;
        return *this;
    }
#define ASSIGN_TYPE(TY, OP)                                         \
    array& array::operator OP(const TY &value)                      \
    {                                                               \
        af::dim4 dims = this->dims();                               \
        af::dtype ty = this->type();                                \
        array cst = constant(value, dims, ty);                      \
        return operator OP(cst);                                    \
    }                                                               \

#define ASSIGN_OP(OP, op1)                                          \
    array& array::operator OP(const array &other)                   \
    {                                                               \
        af_array out = 0;                                           \
        AF_THROW(op1(&out, this->get(), other.get(), gforGet()));   \
        this->set(out);                                             \
        return *this;                                               \
    }                                                               \
    ASSIGN_TYPE(double             , OP)                            \
    ASSIGN_TYPE(float              , OP)                            \
    ASSIGN_TYPE(cdouble            , OP)                            \
    ASSIGN_TYPE(cfloat             , OP)                            \
    ASSIGN_TYPE(int                , OP)                            \
    ASSIGN_TYPE(unsigned           , OP)                            \
    ASSIGN_TYPE(long               , OP)                            \
    ASSIGN_TYPE(unsigned long      , OP)                            \
    ASSIGN_TYPE(long long          , OP)                            \
    ASSIGN_TYPE(unsigned long long , OP)                            \
    ASSIGN_TYPE(char               , OP)                            \
    ASSIGN_TYPE(unsigned char      , OP)                            \
    ASSIGN_TYPE(bool               , OP)                            \
    ASSIGN_TYPE(short              , OP)                            \
    ASSIGN_TYPE(unsigned short     , OP)                            \

    ASSIGN_OP(+=, af_add)
    ASSIGN_OP(-=, af_sub)
    ASSIGN_OP(*=, af_mul)
    ASSIGN_OP(/=, af_div)

#undef ASSIGN_OP

#undef ASSIGN_TYPE

#define ASSIGN_TYPE(TY, OP)                                     \
    array& array::operator OP(const TY &value)                  \
    {                                                           \
        af::dim4 dims = this->dims();                           \
        af::dtype ty = this->type();                            \
        array cst = constant(value, dims, ty);                  \
        return operator OP(cst);                                \
    }                                                           \

#define ASSIGN_OP(OP)                           \
    ASSIGN_TYPE(double             , OP)        \
    ASSIGN_TYPE(float              , OP)        \
    ASSIGN_TYPE(cdouble            , OP)        \
    ASSIGN_TYPE(cfloat             , OP)        \
    ASSIGN_TYPE(int                , OP)        \
    ASSIGN_TYPE(unsigned           , OP)        \
    ASSIGN_TYPE(long               , OP)        \
    ASSIGN_TYPE(unsigned long      , OP)        \
    ASSIGN_TYPE(long long          , OP)        \
    ASSIGN_TYPE(unsigned long long , OP)        \
    ASSIGN_TYPE(char               , OP)        \
    ASSIGN_TYPE(unsigned char      , OP)        \
    ASSIGN_TYPE(bool               , OP)        \
    ASSIGN_TYPE(short              , OP)        \
    ASSIGN_TYPE(unsigned short     , OP)        \

    ASSIGN_OP(= )

#undef ASSIGN_OP

#undef ASSIGN_TYPE

af::dtype implicit_dtype(af::dtype scalar_type, af::dtype array_type)
{
    // If same, do not do anything
    if (scalar_type == array_type) return scalar_type;

    // If complex, return appropriate complex type
    if (scalar_type == c32 || scalar_type == c64) {
        if (array_type == f64 || array_type == c64) return c64;
        return c32;
    }

    // If 64 bit precision, do not lose precision
    if (array_type == f64 || array_type == c64 ||
        array_type == f32 || array_type == c32 ) return array_type;

    // Default to single precision by default when multiplying with scalar
    if ((scalar_type == f64 || scalar_type == c64) &&
        (array_type  != f64 && array_type  != c64)) {
        return f32;
    }

    // Punt to C api for everything else
    return scalar_type;
}

#define BINARY_TYPE(TY, OP, func, dty)                          \
    array operator OP(const array& plhs, const TY &value)       \
    {                                                           \
        af_array out;                                           \
        af::dtype cty = implicit_dtype(dty, plhs.type());       \
        array cst = constant(value, plhs.dims(), cty);          \
        AF_THROW(func(&out, plhs.get(), cst.get(), gforGet())); \
        return array(out);                                      \
    }                                                           \
    array operator OP(const TY &value, const array &other)      \
    {                                                           \
        const af_array rhs = other.get();                       \
        af_array out;                                           \
        af::dtype cty = implicit_dtype(dty, other.type());      \
        array cst = constant(value, other.dims(), cty);         \
        AF_THROW(func(&out, cst.get(), rhs, gforGet()));        \
        return array(out);                                      \
    }                                                           \

#define BINARY_OP(OP, func)                                     \
    array operator OP(const array &lhs, const array &rhs)       \
    {                                                           \
        af_array out;                                           \
        AF_THROW(func(&out, lhs.get(), rhs.get(), gforGet()));  \
        return array(out);                                      \
    }                                                           \
    BINARY_TYPE(double             , OP, func, f64)             \
    BINARY_TYPE(float              , OP, func, f32)             \
    BINARY_TYPE(cdouble            , OP, func, c64)             \
    BINARY_TYPE(cfloat             , OP, func, c32)             \
    BINARY_TYPE(int                , OP, func, s32)             \
    BINARY_TYPE(unsigned           , OP, func, u32)             \
    BINARY_TYPE(long               , OP, func, s64)             \
    BINARY_TYPE(unsigned long      , OP, func, u64)             \
    BINARY_TYPE(long long          , OP, func, s64)             \
    BINARY_TYPE(unsigned long long , OP, func, u64)             \
    BINARY_TYPE(char               , OP, func, b8)              \
    BINARY_TYPE(unsigned char      , OP, func, u8)              \
    BINARY_TYPE(bool               , OP, func, b8)              \
    BINARY_TYPE(short              , OP, func, s16)             \
    BINARY_TYPE(unsigned short     , OP, func, u16)             \

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
    BINARY_OP(%, af_mod)
    BINARY_OP(&, af_bitand)
    BINARY_OP(|, af_bitor)
    BINARY_OP(^, af_bitxor)
    BINARY_OP(<<, af_bitshiftl)
    BINARY_OP(>>, af_bitshiftr)

#undef BINARY_OP

#undef BINARY_TYPE

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
        AF_THROW(af_not(&out, lhs));
        return array(out);
    }

    void array::eval() const
    {
        AF_THROW(af_eval(get()));
    }

// array instanciations
#define INSTANTIATE(T)                                              \
    template<> AFAPI T *array::host() const                         \
    {                                                               \
        if (type() != (af::dtype)dtype_traits<T>::af_type) {        \
            AF_THROW_ERR("Requested type doesn't match with array", \
                         AF_ERR_TYPE);                              \
        }                                                           \
        void *res;                                                  \
        AF_THROW(af_alloc_host(&res, bytes()));                     \
        AF_THROW(af_get_data_ptr(res, get()));                      \
                                                                    \
        return (T*)res;                                             \
    }                                                               \
    template<> AFAPI T array::scalar() const                        \
    {                                                               \
        af_dtype type = (af_dtype)af::dtype_traits<T>::af_type;     \
        if (type != this->type())                                   \
            AF_THROW_ERR("Requested type doesn't match array type", \
                         AF_ERR_TYPE);                              \
        T val;                                                      \
        AF_THROW(af_get_scalar(&val, get()));                       \
        return val;                                                 \
    }                                                               \
    template<> AFAPI T* array::device() const                       \
    {                                                               \
        void *ptr = NULL;                                           \
        AF_THROW(af_get_device_ptr(&ptr, get()));                   \
        return (T *)ptr;                                            \
    }                                                               \
    template<> AFAPI void array::write(const T *ptr,                \
                                       const size_t bytes,          \
                                       af::source src)              \
    {                                                               \
        if(src == afHost)   {                                       \
            AF_THROW(af_write_array(get(), ptr, bytes,              \
                                    (af::source)afHost));           \
        }                                                           \
        if(src == afDevice) {                                       \
            AF_THROW(af_write_array(get(), ptr, bytes,              \
                                    (af::source)afDevice));         \
        }                                                           \
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
    INSTANTIATE(short)
    INSTANTIATE(unsigned short)

#undef INSTANTIATE

    template<> AFAPI void* array::device() const
    {
        void *ptr = NULL;
        AF_THROW(af_get_device_ptr(&ptr, get()));
        return (void *)ptr;
    }


// array_proxy instanciations
#define TEMPLATE_MEM_FUNC(TYPE, RETURN_TYPE, FUNC)      \
    template <> AFAPI                                   \
    RETURN_TYPE array::array_proxy::FUNC() const        \
    {                                                   \
        array out = *this;                              \
        return out.FUNC<TYPE>();                        \
    }

#define INSTANTIATE(T)                                  \
    TEMPLATE_MEM_FUNC(T, T*, host)                      \
    TEMPLATE_MEM_FUNC(T, T , scalar)                    \
    TEMPLATE_MEM_FUNC(T, T*, device)                    \

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
    INSTANTIATE(short)
    INSTANTIATE(unsigned short)

#undef INSTANTIATE
#undef TEMPLATE_MEM_FUNC

    //FIXME: These functions need to be implemented properly at a later point
    void array::array_proxy::unlock() const {}
    void array::array_proxy::lock() const {}
    bool array::array_proxy::isLocked() const { return false; }

    int array::nonzeros() const { return count<int>(*this); }

    void array::lock() const
    {
        AF_THROW(af_lock_array(get()));
    }

    bool array::isLocked() const
    {
        bool res;
        AF_THROW(af_is_locked_array(&res, get()));
        return res;
    }

    void array::unlock() const
    {
        AF_THROW(af_unlock_array(get()));
    }

    void eval(int num, array **arrays)
    {
        std::vector<af_array> outputs(num);
        for (int i = 0; i < num; i++) {
            outputs[i] = arrays[i]->get();
        }
        AF_THROW(af_eval_multiple(num, &outputs[0]));
    }

    void setManualEvalFlag(bool flag)
    {
        AF_THROW(af_set_manual_eval_flag(flag));
    }

    bool getManualEvalFlag()
    {
        bool flag;
        AF_THROW(af_get_manual_eval_flag(&flag));
        return flag;
    }
}
