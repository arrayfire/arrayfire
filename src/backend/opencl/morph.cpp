#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <morph.hpp>
#include <math.hpp>
#include <kernel/morph.hpp>
#include <stdexcept>
#include <iostream>

using af::dim4;

namespace opencl
{

template<typename T>
static Array<T>* morphHelper(kernel::MorphParams &params,
                             const Array<T>      &in)
{
    const dim4 dims     = in.dims();
    const dim4 istrides = in.strides();
    Array<T>* out       = createEmptyArray<T>(dims);
    const dim4 ostrides = out->strides();

    params.offset   = in.getOffset();
    for (dim_type i=0; i<4; ++i) {
        params.dims[i] = dims[i];
        params.istrides[i] = istrides[i];
        params.ostrides[i] = ostrides[i];
    }

    return out;
}

template<typename T, bool isDilation>
Array<T> * morph(const Array<T> &in, const Array<T> &mask)
{
    const dim4 mdims    = mask.dims();

    if (mdims[0]!=mdims[1])
        throw std::runtime_error("Only square masks are supported in cuda morph currently");
    if (mdims[0]>19)
        throw std::runtime_error("Upto 19x19 square kernels are only supported in cuda currently");

    kernel::MorphParams params;
    Array<T>* out       = morphHelper(params, in);

    switch(mdims[0]) {
        case  3: kernel::morph<T, isDilation,  3>(out->get(), in.get(), mask.get(), params); break;
        case  5: kernel::morph<T, isDilation,  5>(out->get(), in.get(), mask.get(), params); break;
        case  7: kernel::morph<T, isDilation,  7>(out->get(), in.get(), mask.get(), params); break;
        case  9: kernel::morph<T, isDilation,  9>(out->get(), in.get(), mask.get(), params); break;
        case 11: kernel::morph<T, isDilation, 11>(out->get(), in.get(), mask.get(), params); break;
        case 13: kernel::morph<T, isDilation, 13>(out->get(), in.get(), mask.get(), params); break;
        case 15: kernel::morph<T, isDilation, 15>(out->get(), in.get(), mask.get(), params); break;
        case 17: kernel::morph<T, isDilation, 17>(out->get(), in.get(), mask.get(), params); break;
        case 19: kernel::morph<T, isDilation, 19>(out->get(), in.get(), mask.get(), params); break;
        default: kernel::morph<T, isDilation,  3>(out->get(), in.get(), mask.get(), params); break;
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

    kernel::MorphParams params;
    Array<T>* out       = morphHelper(params, in);

    switch(mdims[0]) {
        case  3: kernel::morph3d<T, isDilation,  3>(out->get(), in.get(), mask.get(), params); break;
        case  5: kernel::morph3d<T, isDilation,  5>(out->get(), in.get(), mask.get(), params); break;
        case  7: kernel::morph3d<T, isDilation,  7>(out->get(), in.get(), mask.get(), params); break;
        default: kernel::morph3d<T, isDilation,  3>(out->get(), in.get(), mask.get(), params); break;
    }

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
