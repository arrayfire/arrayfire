#include <Array.hpp>
#include <shift.hpp>
#include <kernel/shift.hpp>
#include <stdexcept>
#include <err_cuda.hpp>

namespace cuda
{
    template<typename T>
    Array<T> *shift(const Array<T> &in, const af::dim4 &sdims)
    {
        const af::dim4 iDims = in.dims();
        af::dim4 oDims = iDims;

        Array<T> *out = createEmptyArray<T>(oDims);

        kernel::shift<T>(*out, in, sdims.get());

        return out;
    }

#define INSTANTIATE(T)                                                          \
    template Array<T>* shift<T>(const Array<T> &in, const af::dim4 &sdims);     \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)
}
