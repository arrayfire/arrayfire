#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <morph.hpp>
#include <kernel/morph.hpp>
#include <stdexcept>

using af::dim4;

namespace cuda
{

template<typename T, bool isDilation>
Array<T> * morph(const Array<T> &in, const Array<T> &mask)
{
    const dim4 mdims = mask.dims();

    if (mdims[0] != mdims[1])
        throw std::runtime_error("Only square masks are supported in cuda morph currently");
    if (mdims[0] > 19)
        throw std::runtime_error("Upto 19x19 square kernels are only supported in cuda currently");

    Array<T>* out = createEmptyArray<T>(in.dims());

    cudaMemcpyToSymbol(kernel::cFilter, mask.get(),
                       mdims[0] * mdims[1] * sizeof(T),
                       0, cudaMemcpyDeviceToDevice);

    if (isDilation)
        kernel::morph<T, true >(*out, in, mdims[0]);
    else
        kernel::morph<T, false>(*out, in, mdims[0]);

    return out;
}

template<typename T, bool isDilation>
Array<T> * morph3d(const Array<T> &in, const Array<T> &mask)
{
    const dim4 mdims = mask.dims();

    if (mdims[0] != mdims[1] || mdims[0] != mdims[2])
        throw std::runtime_error("Only cube masks are supported in cuda morph currently");
    if (mdims[0] > 7)
        throw std::runtime_error("Upto 7x7x7 kernels are only supported in cuda currently");

    if (in.dims()[3] > 1)
        throw std::runtime_error("Batch not supported for volumetic morphological operations");

    Array<T>* out       = createEmptyArray<T>(in.dims());

    cudaMemcpyToSymbol(kernel::cFilter, mask.get(),
                       mdims[0] * mdims[1] *mdims[2] * sizeof(T),
                       0, cudaMemcpyDeviceToDevice);

    if (isDilation)
        kernel::morph3d<T, true >(*out, in, mdims[0]);
    else
        kernel::morph3d<T, false>(*out, in, mdims[0]);

    return out;
}

#define INSTANTIATE(T)\
    template Array<T> * morph  <T, true >(const Array<T> &in, const Array<T> &mask);\
    template Array<T> * morph  <T, false>(const Array<T> &in, const Array<T> &mask);\
    template Array<T> * morph3d<T, true >(const Array<T> &in, const Array<T> &mask);\
    template Array<T> * morph3d<T, false>(const Array<T> &in, const Array<T> &mask);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(char  )
INSTANTIATE(int   )
INSTANTIATE(uint  )
INSTANTIATE(uchar )

}
