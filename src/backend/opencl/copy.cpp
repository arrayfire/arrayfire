#define __CL_ENABLE_EXCEPTIONS
#include <iostream>
#include <cassert>
#include <iterator>
#include <af/array.h>
#include <af/defines.h>
#include <Array.hpp>
#include <copy.hpp>

namespace opencl
{

template<typename T>
void copyData(T *data, const af_array &arr)
{
    const Array<T> &val_impl = getArray<T>(arr);
    //FIXME: Add checks
    cl::copy(getQueue(0), val_impl.get(), data, data + val_impl.elements());
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
