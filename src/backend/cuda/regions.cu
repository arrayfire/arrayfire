#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <regions.hpp>
#include <kernel/regions.hpp>
#include <err_cuda.hpp>

using af::dim4;

namespace cuda
{

template<typename T>
Array<T> * regions(const Array<uchar> &in, const unsigned connectivity)
{
    ARG_ASSERT(2, (connectivity==4 || connectivity==8));

    const dim4 dims = in.dims();

    Array<T> * out  = createEmptyArray<T>(dims);

    // Create bindless texture object for the equiv map.
    cudaTextureObject_t tex = 0;
    // FIXME: Currently disabled, only supported on capaibility >= 3.0
    //if (compute >= 3.0) {
    //    cudaResourceDesc resDesc;
    //    memset(&resDesc, 0, sizeof(resDesc));
    //    resDesc.resType = cudaResourceTypeLinear;
    //    resDesc.res.linear.devPtr = out->get();
    //    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    //    resDesc.res.linear.desc.x = 32; // bits per channel
    //    resDesc.res.linear.sizeInBytes = dims[0] * dims[1] * sizeof(float);
    //    cudaTextureDesc texDesc;
    //    memset(&texDesc, 0, sizeof(texDesc));
    //    texDesc.readMode = cudaReadModeElementType;
    //    CUDA_CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
    //}

    switch(connectivity) {
        case 4:
            regions<T, false>(*out, in, tex);
            break;
        case 8:
            regions<T, true >(*out, in, tex);
            break;
    }

    return out;
}

#define INSTANTIATE(T)\
    template Array<T> * regions<T>(const Array<uchar> &in, const unsigned connectivity);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(char  )
INSTANTIATE(int   )
INSTANTIATE(uint  )
INSTANTIATE(uchar )

}
