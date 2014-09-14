#include <Array.hpp>
#include <tile.hpp>
#include <kernel/tile.hpp>
#include <stdexcept>
#include <err_cuda.hpp>

namespace cuda
{
    template<typename T>
    Array<T> *tile(const Array<T> &in, const af::dim4 &tileDims)
    {
        const af::dim4 iDims = in.dims();
        af::dim4 oDims = iDims;
        oDims *= tileDims;

        if(iDims.elements() == 0 || oDims.elements() == 0) {
            AF_ERROR("Elements are 0", AF_ERR_SIZE);
        }

        Array<T> *out = createEmptyArray<T>(oDims);

        kernel::tile<T>(*out, in);

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
