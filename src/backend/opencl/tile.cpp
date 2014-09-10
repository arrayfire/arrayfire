#include <Array.hpp>
#include <tile.hpp>
#include <kernel/tile.hpp>
#include <stdexcept>

namespace opencl
{
    template<typename T>
    Array<T> *tile(const Array<T> &in, const af::dim4 &tileDims)
    {
        const af::dim4 iDims = in.dims();
        af::dim4 oDims = iDims;
        oDims *= tileDims;

        if(iDims.elements() == 0 || oDims.elements() == 0) {
            throw std::runtime_error("Elements are 0");
        }

        Array<T> *out = createEmptyArray<T>(oDims);

        kernel::tile<T>(out->get(), in.get(), out->dims().get(), in.dims().get(),
                     out->strides().get(), in.strides().get(), in.getOffset());

        return out;
    }

#define INSTANTIATE(T)                                                         \
    template Array<T>* tile<T>(const Array<T> &in, const af::dim4 &tileDims);  \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)

}

