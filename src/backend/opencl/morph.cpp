#define __CL_ENABLE_EXCEPTIONS
#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <morph.hpp>
#include <kernel/morph.hpp>
#include <stdexcept>
#include <iostream>

using af::dim4;

namespace opencl
{

template<typename T, bool isDilation>
Array<T> * morph(const Array<T> &in, const Array<T> &mask)
{
    const dim4 mdims    = mask.dims();

    if (mdims[0]!=mdims[1])
        throw std::runtime_error("Only square masks are supported in cuda morph currently");
    if (mdims[0]>17)
        throw std::runtime_error("Upto 17x17 square kernels are only supported in cuda currently");

    const dim4 dims     = in.dims();
    const dim4 istrides = in.strides();
    Array<T>* out       = createEmptyArray<T>(dims);
    const dim4 ostrides = out->strides();

    kernel::MorphParams params;
    params.windLen  = mdims[0];
    params.dim0  = dims[0];
    params.dim1  = dims[1];
    params.dim2  = dims[2];
    params.offset   = in.getOffset();
    params.istride0 = istrides[0];
    params.istride1 = istrides[1];
    params.istride2 = istrides[2];
    params.istride3 = istrides[3];
    params.ostride0 = ostrides[0];
    params.ostride1 = ostrides[1];
    params.ostride2 = ostrides[2];
    params.ostride3 = ostrides[3];

    try {
        if (isDilation)
            kernel::morph<T, true>(out->get(), in.get(), mask.get(), params);
        else
            kernel::morph<T, false>(out->get(), in.get(), mask.get(), params);
    }catch(cl::Error error) {
        throw std::runtime_error(std::string("@opencl/morph: ").append(error.what()));
    }

    return out;
}

template<typename T, bool isDilation>
Array<T> * morph3d(const Array<T> &in, const Array<T> &mask)
{
    const dim4 mdims    = mask.dims();

    if (mdims[0]!=mdims[1] || mdims[0]!=mdims[2])
        throw std::runtime_error("Only cube masks are supported in cuda morph currently");
    if (mdims[0]>7)
        throw std::runtime_error("Upto 7x7x7 kernels are only supported in cuda currently");

    const dim4 dims     = in.dims();
    if (dims[3]>1)
        throw std::runtime_error("Batch not supported for volumetic morphological operations");

    Array<T>* out       = createEmptyArray<T>(dims);

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
