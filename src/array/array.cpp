#include <af/array.h>
#include <af/traits.hpp>
#include <ArrayInfo.hpp>
#include <af/index.h>
#include "error.hpp"

namespace af
{
    array::array() : arr(0) {}
    array::array(const af_array handle): arr(handle) {}

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
        if (src != afHost) AF_THROW(AF_ERR_INVALID_ARG);

        dim_type my_dims[] = {d0, d1, d2, d3};
        AF_THROW(af_create_array(arr, (const void * const)ptr, 4, my_dims, ty));
    }

    array::array(const dim4 &dims, af_dtype ty) : arr(0)
    {
        initEmptyArray(&arr, ty, dims[0], dims[1], dims[2], dims[3]);
    }

    array::array(dim_type d0, af_dtype ty) : arr(0)
    {
        initEmptyArray(&arr, ty, d0);
    }

    array::array(dim_type d0, dim_type d1, af_dtype ty) : arr(0)
    {
        initEmptyArray(&arr, ty, d0, d1);
    }

    array::array(dim_type d0, dim_type d1, dim_type d2, af_dtype ty) : arr(0)
    {
        initEmptyArray(&arr, ty, d0, d1, d2);
    }

    array::array(dim_type d0, dim_type d1, dim_type d2, dim_type d3, af_dtype ty) : arr(0)
    {
        initEmptyArray(&arr, ty, d0, d1, d2, d3);
    }

    template<typename T>
    array::array(const dim4 &dims, const T *ptr, af_source_t src, dim_type ngfor) : arr(0)
    {
        initDataArray<T>(&arr, ptr, src, dims[0], dims[1], dims[2], dims[3]);
    }

    template<typename T>
    array::array(dim_type d0, const T *ptr, af_source_t src, dim_type ngfor) : arr(0)
    {
        initDataArray<T>(&arr, ptr, src, d0);
    }

    template<typename T>
    array::array(dim_type d0, dim_type d1, const T *ptr,
                 af_source_t src, dim_type ngfor) : arr(0)
    {
        initDataArray<T>(&arr, ptr, src, d0, d1);
    }

    template<typename T>
    array::array(dim_type d0, dim_type d1, dim_type d2, const T *ptr,
                 af_source_t src, dim_type ngfor) : arr(0)
    {
        initDataArray<T>(&arr, ptr, src, d0, d1, d2);
    }

    template<typename T>
    array::array(dim_type d0, dim_type d1, dim_type d2, dim_type d3,
                 const T *ptr, af_source_t src, dim_type ngfor) : arr(0)
    {
        initDataArray<T>(&arr, ptr, src, d0, d1, d2, d3);
    }

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

    template<typename T>
    T *array::host() const
    {
        if (type() != (af_dtype)dtype_traits<T>::af_type) {
            AF_THROW(AF_ERR_INVALID_TYPE);
        }

        T *res = new T[elements()];
        AF_THROW(af_get_data_ptr((void *)res, arr));

        return res;
    }

    void array::host(void *data) const
    {
        AF_THROW(af_get_data_ptr(data, arr));
    }

    af_array array::get() const
    {
        return arr;
    }

    // Helper functions
    dim4 array::dims() const
    {
        ArrayInfo info = getInfo(arr);
        return info.dims();
    }

    dim_type array::dims(unsigned dim) const
    {
        ArrayInfo info = getInfo(arr);
        return info.dims()[dim];
    }

    unsigned array::numdims() const
    {
        ArrayInfo info = getInfo(arr);
        return info.ndims();
    }

    size_t array::bytes() const
    {
        ArrayInfo info = getInfo(arr);
        return info.elements() * sizeof(type());
    }

    array array::copy() const
    {
        af_array *other = 0;
        AF_THROW(af_copy_array(other, arr));
        return array(*other);
    }

    bool array::isempty() const
    {
        ArrayInfo info = getInfo(arr);
        return info.isEmpty();
    }

    bool array::isscalar() const
    {
        ArrayInfo info = getInfo(arr);
        return info.isScalar();
    }

    bool array::isvector() const
    {
        ArrayInfo info = getInfo(arr);
        return info.isVector();
    }

    bool array::isrow() const
    {
        ArrayInfo info = getInfo(arr);
        return info.isRow();
    }

    bool array::iscolumn() const
    {
        ArrayInfo info = getInfo(arr);
        return info.isColumn();
    }

    bool array::iscomplex() const
    {
        ArrayInfo info = getInfo(arr);
        return info.isComplex();
    }

    bool array::isdouble() const
    {
        ArrayInfo info = getInfo(arr);
        return info.isDouble();
    }

    bool array::issingle() const
    {
        ArrayInfo info = getInfo(arr);
        return info.isSingle();
    }

    bool array::isrealfloating() const
    {
        ArrayInfo info = getInfo(arr);
        return info.isRealFloating();
    }

    bool array::isfloating() const
    {
        ArrayInfo info = getInfo(arr);
        return info.isFloating();
    }

    bool array::isinteger() const
    {
        ArrayInfo info = getInfo(arr);
        return info.isInteger();
    }

    array constant(double val, const dim4 &dims, af_dtype type)
    {
        af_array res;
        AF_THROW(af_constant(&res, val, dims.ndims(), dims.get(), type));
        return array(res);
    }

    array constant(double val, const dim_type d0, af_dtype ty)
    {
        return constant(val, dim4(d0), ty);
    }

    array constant(double val, const dim_type d0,
                         const dim_type d1, af_dtype ty)
    {
        return constant(val, dim4(d0, d1), ty);
    }

    array constant(double val, const dim_type d0,
                         const dim_type d1, const dim_type d2, af_dtype ty)
    {
        return constant(val, dim4(d0, d1, d2), ty);
    }

    array constant(double val, const dim_type d0,
                         const dim_type d1, const dim_type d2,
                         const dim_type d3, af_dtype ty)
    {
        return constant(val, dim4(d0, d1, d2, d3), ty);
    }


    array randu(const dim4 &dims, af_dtype type)
    {
        af_array res;
        AF_THROW(af_randu(&res, dims.ndims(), dims.get(), type));
        return array(res);
    }

    array randu(const dim_type d0, af_dtype ty)
    {
        return randu(dim4(d0), ty);
    }

    array randu(const dim_type d0,
                const dim_type d1, af_dtype ty)
    {
        return randu(dim4(d0, d1), ty);
    }

    array randu(const dim_type d0,
                const dim_type d1, const dim_type d2, af_dtype ty)
    {
        return randu(dim4(d0, d1, d2), ty);
    }

    array randu(const dim_type d0,
                const dim_type d1, const dim_type d2,
                const dim_type d3, af_dtype ty)
    {
        return randu(dim4(d0, d1, d2, d3), ty);
    }

    array randn(const dim4 &dims, af_dtype type)
    {
        af_array res;
        AF_THROW(af_randn(&res, dims.ndims(), dims.get(), type));
        return array(res);
    }

    array randn(const dim_type d0, af_dtype ty)
    {
        return randn(dim4(d0), ty);
    }

    array randn(const dim_type d0,
                const dim_type d1, af_dtype ty)
    {
        return randn(dim4(d0, d1), ty);
    }

    array randn(const dim_type d0,
                const dim_type d1, const dim_type d2, af_dtype ty)
    {
        return randn(dim4(d0, d1, d2), ty);
    }

    array randn(const dim_type d0,
                const dim_type d1, const dim_type d2,
                const dim_type d3, af_dtype ty)
    {
        return randn(dim4(d0, d1, d2, d3), ty);
    }

    array array::operator()(const af_seq& s0, const af_seq& s1, const af_seq& s2, const af_seq& s3)
    {
        af_array out = 0;
        af_seq indices[] = {s0, s1, s2, s3};
        AF_THROW(af_index(&out, this->get(), 4, indices));
        return array(out);
    }

    array& array::operator=(const array &other)
    {
        // FIXME
        if (this->get() == other.get()) {
            return *this;
        }
        if(this->get() != 0) {
            AF_THROW(af_destroy_array(this->get()));
        }

        af_array temp = 0;
        AF_THROW(af_weak_copy(&temp, other.get()));
        this->arr = temp;
        return *this;
    }

    array::array(const array &in) : arr(0)
    {
        AF_THROW(af_weak_copy(&arr, in.get()));
    }

#define INSTANTIATE(T)  \
    template array::array<T>(const dim4 &dims, const T *ptr, af_source_t src, dim_type ngfor);\
    template array::array<T>(dim_type d0, const T *ptr, af_source_t src, dim_type ngfor);\
    template array::array<T>(dim_type d0, dim_type d1, const T *ptr, af_source_t src, dim_type ngfor);\
    template array::array<T>(dim_type d0, dim_type d1, dim_type d2, const T *ptr, af_source_t src, dim_type ngfor);\
    template array::array<T>(dim_type d0, dim_type d1, dim_type d2, dim_type d3, const T *ptr, af_source_t src, dim_type ngfor);\
    template T *array::host<T>() const;


    INSTANTIATE(af_cdouble)
    INSTANTIATE(af_cfloat)
    INSTANTIATE(double)
    INSTANTIATE(float)
    INSTANTIATE(unsigned)
    INSTANTIATE(int)
    INSTANTIATE(unsigned char)
    INSTANTIATE(char)
}
