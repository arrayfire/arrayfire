#include <cassert>

#include <cuda_runtime_api.h>
#include <af/array.h>
#include <af/defines.h>
#include <copy.hpp>
#include <Array.hpp>

namespace cuda
{

template<typename T>
T* copyData(const af_array &arr)
{
    size_t elements = af_get_elements(arr);
    const Array<T> &val_arr = getArray<T>(arr);
    // can't use vector because we need to release ownership
    T* out = new T[elements];

    //FIXME: Add checks
    cudaMemcpy(out, val_arr.get(), elements*sizeof(T), cudaMemcpyDeviceToHost);

    return out;
}

template<typename T>
void
copyData(af_array &dst, const T* const src)
{
    Array<T> &dstArray = getWritableArray<T>(dst);
    if(dstArray.isOwner()) {
        cudaMemcpy(dstArray.get(), src, dstArray.elements() * sizeof(T), cudaMemcpyHostToDevice);
    }
    else {
        assert("NOT IMPLEMENTED" && 1 != 1);
    }
}


template void copyData<float>(af_array &dst, const float* const src);
template void copyData<cfloat>(af_array &dst, const cfloat* const src);
template void copyData<double>(af_array &dst, const double* const src);
template void copyData<cdouble>(af_array &dst, const cdouble* const src);
template void copyData<char>(af_array &dst, const char* const src);
template void copyData<int>(af_array &dst, const int* const src);
template void copyData<unsigned>(af_array &dst, const unsigned* const src);
template void copyData<unsigned char>(af_array &dst, const unsigned char* const src);

template float*                             copyData<float>(const af_array &arr);
template cfloat*                            copyData<cfloat>(const af_array &arr);
template double*                            copyData<double>(const af_array &arr);
template cdouble*                           copyData<cdouble>(const af_array &arr);
template char*                              copyData<char>(const af_array &arr);
template int*                               copyData<int>(const af_array &arr);
template unsigned*                          copyData<unsigned>(const af_array &arr);
template unsigned char*                     copyData<unsigned char>(const af_array &arr);
}
