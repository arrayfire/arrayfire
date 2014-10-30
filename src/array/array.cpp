#include <af/array.h>
#include <af/arith.h>
#include <af/traits.hpp>
#include <ArrayInfo.hpp>
#include <af/index.h>
#include "error.hpp"

namespace af
{
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
        if (src != afHost) AF_THROW(AF_ERR_INVALID_ARG);

        dim_type my_dims[] = {d0, d1, d2, d3};
        AF_THROW(af_create_array(arr, (const void * const)ptr, 4, my_dims, ty));
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

    template<typename T>
    array::array(const dim4 &dims, const T *ptr, af_source_t src, dim_type ngfor) : arr(0), isRef(false)
    {
        initDataArray<T>(&arr, ptr, src, dims[0], dims[1], dims[2], dims[3]);
    }

    template<typename T>
    array::array(dim_type d0, const T *ptr, af_source_t src, dim_type ngfor) : arr(0), isRef(false)
    {
        initDataArray<T>(&arr, ptr, src, d0);
    }

    template<typename T>
    array::array(dim_type d0, dim_type d1, const T *ptr,
                 af_source_t src, dim_type ngfor) : arr(0), isRef(false)
    {
        initDataArray<T>(&arr, ptr, src, d0, d1);
    }

    template<typename T>
    array::array(dim_type d0, dim_type d1, dim_type d2, const T *ptr,
                 af_source_t src, dim_type ngfor) : arr(0), isRef(false)
    {
        initDataArray<T>(&arr, ptr, src, d0, d1, d2);
    }

    template<typename T>
    array::array(dim_type d0, dim_type d1, dim_type d2, dim_type d3,
                 const T *ptr, af_source_t src, dim_type ngfor) : arr(0), isRef(false)
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

    af_array array::get()
    {
        if (!isRef)
            return arr;
        af_array temp = 0;
        AF_THROW(af_index(&temp, arr, 4, this->s));
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

    array::array(af_array in, af_seq *seqs) : arr(in), isRef(true)
    {
        for(int i=0; i<4; ++i) s[i] = seqs[i];
    }

    array array::operator()(const af_seq& s0, const af_seq& s1, const af_seq& s2, const af_seq& s3) const
    {
        af_array out = 0;
        af_seq indices[] = {s0, s1, s2, s3};
        //FIXME: check if this->s has same dimensions as numdims
        AF_THROW(af_weak_copy(&out, this->get()));
        return array(out, indices);
    }
    array array::row(size_t index) const
    {
        af_seq idx = {index, index, 1};
        return this->operator()(idx, span, span, span);
    }

    array array::col(size_t index) const
    {
        af_seq idx = {index, index, 1};
        return this->operator()(span, idx, span, span);
    }

    array array::slice(size_t index) const
    {
        af_seq idx = {index, index, 1};
        return this->operator()(span, span, idx, span);
    }

    array array::rows(size_t first, size_t last) const
    {
        af_seq idx = {first, last, 1};
        return this->operator()(idx, span, span, span);
    }

    array array::cols(size_t first, size_t last) const
    {
        af_seq idx = {first, last, 1};
        return this->operator()(span, idx, span, span);
    }

    array array::slices(size_t first, size_t last) const
    {
        af_seq idx = {first, last, 1};
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

#undef INSTANTIATE

}
