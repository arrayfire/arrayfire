#include <af/dim4.hpp>
#include <Array.hpp>
#include <stdexcept>
#include <copy.hpp>
#include <kernel/elwise.hpp> //set
#include <err_cuda.hpp>

using af::dim4;

namespace cuda
{
    using std::ostream;

    template<typename T>
    T* cudaMallocWrapper(const size_t &elements)
    {
        T* ptr = NULL;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * elements));
        return ptr;
    }

    template<typename T>
    Array<T>::Array(af::dim4 dims) :
        ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(cudaMallocWrapper<T>(dims.elements())),
        parent()
    {}

    template<typename T>
    Array<T>::Array(af::dim4 dims, T val) :
        ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(cudaMallocWrapper<T>(dims.elements())),
        parent()
    {
        kernel::set(data, val, elements());
    }

    template<typename T>
    Array<T>::Array(af::dim4 dims, const T * const in_data) :
    ArrayInfo(dims, af::dim4(0,0,0,0), calcStrides(dims), (af_dtype)dtype_traits<T>::af_type),
        data(cudaMallocWrapper<T>(dims.elements())),
        parent()
    {
        CUDA_CHECK(cudaMemcpy(data, in_data, dims.elements() * sizeof(T), cudaMemcpyHostToDevice));
    }

    template<typename T>
    Array<T>::Array(const Array<T>& parnt, const dim4 &dims, const dim4 &offset, const dim4 &stride) :
        ArrayInfo(dims, offset, stride, (af_dtype)dtype_traits<T>::af_type),
        data(0),
        parent(&parnt)
    { }

    template<typename T>
    Array<T>::Array(Param<T> &tmp) :
        ArrayInfo(af::dim4(tmp.dims[0], tmp.dims[1], tmp.dims[2], tmp.dims[3]),
                  af::dim4(0, 0, 0, 0),
                  af::dim4(tmp.strides[0], tmp.strides[1], tmp.strides[2], tmp.strides[3]),
                  (af_dtype)dtype_traits<T>::af_type),
        data(tmp.ptr),
        parent()
    {
    }

    template<typename T>
    Array<T>::~Array() { CUDA_CHECK(cudaFree(data)); }

    template<typename T>
    Array<T> *
    createDataArray(const dim4 &size, const T * const data)
    {
        Array<T> *out = new Array<T>(size, data);
        return out;
    }

    template<typename T>
    Array<T>*
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

        // FIXME: Implement this for CUDA
        if (stride[0] != 1 ||
            stride[1] <  0 ||
            stride[2] <  0 ||
            stride[3] <  0) {
            out = copyArray(*out);
        }

        return out;
    }

    template<typename T>
    Array<T>*
    createParamArray(Param<T> &tmp)
    {
        Array<T> *out = new Array<T>(tmp);
        return out;
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
    template       Array<T>*  createParamArray<T> (Param<T> &tmp);           \
    template       Array<T>*  createSubArray<T>       (const Array<T> &parent, const dim4 &dims, const dim4 &offset, const dim4 &stride); \
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
}
