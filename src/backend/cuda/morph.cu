#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <morph.hpp>
#include <kernel/morph.hpp>
#include <stdexcept>

#include <cstdio>

using af::dim4;

namespace cuda
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

    kernel::morph_param_t<T> params;
    params.d_dst    = out->get();
    params.d_src    = in.get();
    params.windLen  = mdims[0];

    cudaMemcpyToSymbol(kernel::cFilter, mask.get(),
            mdims[0]*mdims[1]*sizeof(T), 0, cudaMemcpyDeviceToDevice);

    for (int i=0;i<4; ++i) {
        params.dims[i]     = dims[i];
        params.istrides[i] = istrides[i];
        params.ostrides[i] = ostrides[i];
    }

    if (isDilation)
        kernel::morph<T, true>(params);
    else
        kernel::morph<T, false>(params);

    return out;
}

template<typename T, bool isDilation>
Array<T> * morph3d(const Array<T> &in, const Array<T> &mask)
{
    const dim4 dims   = in.dims();

    Array<T>* out     = createEmptyArray<T>(dims);

    T* outData        = out->get();
    const T*   inData = in.get();

    throw std::runtime_error("3D Morphological operations are not supported in cuda yet");

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
