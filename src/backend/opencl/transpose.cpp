#include <af/dim4.hpp>
#include <Array.hpp>
#include <transpose.hpp>
#include <kernel/transpose.hpp>

using af::dim4;

namespace opencl
{

    template<typename T>
    Array<T> * transpose(const Array<T> &in)
    {
        const dim4 inDims   = in.dims();
        const dim4 inStrides= in.strides();

        dim4 outDims  = dim4(inDims[1],inDims[0],inDims[2],inDims[3]);

        Array<T>* out  = createEmptyArray<T>(outDims);

        kernel::transpose<T>(out->get(), in.get(), inDims.ndims(), inDims.get(), inStrides.get(), in.getOffset());

        return out;
    }

#define INSTANTIATE(T)\
    template Array<T> * transpose(const Array<T> &in);

    INSTANTIATE(float)
    INSTANTIATE(cfloat)
    INSTANTIATE(double)
    INSTANTIATE(cdouble)
    INSTANTIATE(char)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)

}
