#include <cassert>

#include <cuda_runtime_api.h>
#include <af/array.h>
#include <af/defines.h>
#include <Array.hpp>
#include <copy.hpp>

namespace cuda
{

template<typename T>
void copyData(T *data, const af_array &arr)
{
    const Array<T> &val_arr = getArray<T>(arr);

    //FIXME: Add checks
    cudaMemcpy(data, val_arr.get(), val_arr.elements()*sizeof(T), cudaMemcpyDeviceToHost);

    return;
}

template void copyData<float>(float *data, const af_array &dst);
template void copyData<cfloat>(cfloat *data, const af_array &dst);
template void copyData<double>(double *data, const af_array &dst);
template void copyData<cdouble>(cdouble *data, const af_array &dst);
template void copyData<char>(char *data, const af_array &dst);
template void copyData<int>(int *data, const af_array &dst);
template void copyData<unsigned>(unsigned *data, const af_array &dst);
template void copyData<uchar>(uchar *data, const af_array &dst);

}
