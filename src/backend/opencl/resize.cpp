#include <af/array.h>
#include <af/dim4.hpp>
#include <Array.hpp>
#include <resize.hpp>
#include <stdexcept>

namespace opencl
{
    template<typename T>
    Array<T>* resize(const Array<T> &in, const dim_type odim0, const dim_type odim1,
                     const af_interp_type method)
    {
        throw std::runtime_error("Resize not supported in OpenCL yet.");
        Array<T> *outArray;
        return outArray;
    }


#define INSTANTIATE(T)                                                                            \
    template Array<T>* resize<T> (const Array<T> &in, const dim_type odim0, const dim_type odim1, \
                                  const af_interp_type method);


    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)
}
