#include <Array.hpp>
#include <reorder.hpp>
#include <kernel/reorder.hpp>
#include <stdexcept>
#include <err_opencl.hpp>

namespace opencl
{
    template<typename T>
    Array<T> *reorder(const Array<T> &in, const af::dim4 &rdims)
    {
        const af::dim4 iDims = in.dims();
        af::dim4 oDims(0);
        for(int i = 0; i < 4; i++)
            oDims[i] = iDims[rdims[i]];

        Array<T> *out = createEmptyArray<T>(oDims);

        kernel::reorder<T>(*out, in, rdims.get());

        return out;
    }

#define INSTANTIATE(T)                                                         \
    template Array<T>* reorder<T>(const Array<T> &in, const af::dim4 &rdims);  \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)
}

