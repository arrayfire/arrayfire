#include <af/dim4.hpp>
#include <Array.hpp>
#include <copy.hpp>
#include <iostream>

namespace cpu
{

    using af::dim4;

    template<typename T>
    Array<T>::Array(dim4 dims):
        ArrayInfo(dims, dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(dims.elements()),
        parent(nullptr)
    { }

    template<typename T>
    Array<T>::Array(dim4 dims, T val):
        ArrayInfo(dims, dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(dims.elements(), val),
        parent(nullptr)
    { }

    template<typename T>
    Array<T>::Array(dim4 dims, const T * const in_data):
        ArrayInfo(dims, dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(in_data, in_data + dims.elements()),
        parent(nullptr)
    { }

    template<typename T>
    Array<T>::Array(const Array<T>& parnt, const dim4 &dims, const dim4 &offset, const dim4 &stride) :
        ArrayInfo(dims, offset, stride, (af_dtype)dtype_traits<T>::af_type),
        data(0),
        parent(&parnt)
    { }

    template<typename T>
    Array<T>::~Array()
    { }

    template<typename T>
    Array<T> *
    createDataArray(const dim4 &size, const T * const data)
    {
        Array<T> *out = new Array<T>(size, data);
        return out;
    }

    template<typename T>
    Array<T> *
    createValueArray(const dim4 &size, const T& value)
    {
        Array<T> *out = new Array<T>(size, value);
        return out;
    }

    template<typename T>
    Array<T>*
    createEmptyArray(const dim4 &size)
    {
        Array<T> *out = new Array<T>(size);
        return out;
    }

    template<typename T>
    Array<T> *
    createSubArray(const Array<T>& parent, const dim4 &dims, const dim4 &offset, const dim4 &stride)
    {
        Array<T> *out = new Array<T>(parent, dims, offset, stride);
        // FIXME: check what is happening with the references here
        if (stride[0] != 1 ||
            stride[1] <  0 ||
            stride[2] <  0 ||
            stride[3] <  0) out = copyArray(*out);
        return out;
    }

    template<typename T>
    Array<T> *
    createRefArray(const Array<T>& parent, const dim4 &dims, const dim4 &offset, const dim4 &stride)
    {
        return new Array<T>(parent, dims, offset, stride);
    }

    template<typename inType, typename outType>
    Array<outType> *
    createPaddedArray(Array<inType> const &in, dim4 const &dims, outType default_value)
    {
        Array<outType> *ret = createValueArray<outType>(dims, default_value);

        copy<inType, outType>(*ret, in, outType(default_value), 1.0);

        return ret;
    }

    template<typename T>
    void scaleArray(Array<T> &arr, double factor)
    {
        T * src_ptr = arr.get();
        for(dim_type i=0; i< (dim_type)arr.elements(); ++i)
            src_ptr[i] *= factor;
    }

    template<typename T>
    void
    destroyArray(Array<T> &A)
    {
        delete &A;
    }

#define INSTANTIATE(T)                                                  \
    template       Array<T>*  createDataArray<T>  (const dim4 &size, const T * const data); \
    template       Array<T>*  createValueArray<T> (const dim4 &size, const T &value); \
    template       Array<T>*  createEmptyArray<T> (const dim4 &size);   \
    template       Array<T>*  createSubArray<T>   (const Array<T> &parent, const dim4 &dims, const dim4 &offset, const dim4 &stride); \
    template       Array<T>*  createRefArray<T>   (const Array<T> &parent, const dim4 &dims, const dim4 &offset, const dim4 &stride); \
    template       void       scaleArray<T>       (Array<T> &arr, double factor); \
    template       void       destroyArray<T>     (Array<T> &A);        \
    template                  Array<T>::~Array();

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)

#define INSTANTIATE_CREATE_PADDED_ARRAY(SRC_T) \
    template Array<float  >* createPaddedArray<SRC_T, float  >(Array<SRC_T> const &src, dim4 const &dims, float   default_value); \
    template Array<double >* createPaddedArray<SRC_T, double >(Array<SRC_T> const &src, dim4 const &dims, double  default_value); \
    template Array<cfloat >* createPaddedArray<SRC_T, cfloat >(Array<SRC_T> const &src, dim4 const &dims, cfloat  default_value); \
    template Array<cdouble>* createPaddedArray<SRC_T, cdouble>(Array<SRC_T> const &src, dim4 const &dims, cdouble default_value); \
    template Array<int    >* createPaddedArray<SRC_T, int    >(Array<SRC_T> const &src, dim4 const &dims, int     default_value); \
    template Array<uint   >* createPaddedArray<SRC_T, uint   >(Array<SRC_T> const &src, dim4 const &dims, uint    default_value); \
    template Array<uchar  >* createPaddedArray<SRC_T, uchar  >(Array<SRC_T> const &src, dim4 const &dims, uchar   default_value); \
    template Array<char   >* createPaddedArray<SRC_T, char   >(Array<SRC_T> const &src, dim4 const &dims, char    default_value);

    INSTANTIATE_CREATE_PADDED_ARRAY(float )
    INSTANTIATE_CREATE_PADDED_ARRAY(double)
    INSTANTIATE_CREATE_PADDED_ARRAY(int   )
    INSTANTIATE_CREATE_PADDED_ARRAY(uint  )
    INSTANTIATE_CREATE_PADDED_ARRAY(uchar )
    INSTANTIATE_CREATE_PADDED_ARRAY(char  )

#define INSTANTIATE_CREATE_COMPLEX_PADDED_ARRAY(SRC_T) \
    template Array<cfloat >* createPaddedArray<SRC_T, cfloat >(Array<SRC_T> const &src, dim4 const &dims, cfloat  default_value); \
    template Array<cdouble>* createPaddedArray<SRC_T, cdouble>(Array<SRC_T> const &src, dim4 const &dims, cdouble default_value);

    INSTANTIATE_CREATE_COMPLEX_PADDED_ARRAY(cfloat )
    INSTANTIATE_CREATE_COMPLEX_PADDED_ARRAY(cdouble)

}
