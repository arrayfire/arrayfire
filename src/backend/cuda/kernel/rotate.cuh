/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Param.hpp>
#include <interp.hpp>

namespace arrayfire {
namespace cuda {

typedef struct {
    float tmat[6];
} tmat_t;

template<typename T, int order>
__global__ void rotate(Param<T> out, CParam<T> in, const tmat_t t,
                       const int nimages, const int nbatches,
                       const int blocksXPerImage, const int blocksYPerImage,
                       af::interpType method) {
    // Compute which image set
    const int setId      = blockIdx.x / blocksXPerImage;
    const int blockIdx_x = blockIdx.x - setId * blocksXPerImage;

    const int batch      = blockIdx.y / blocksYPerImage;
    const int blockIdx_y = blockIdx.y - batch * blocksYPerImage;

    // Get thread indices
    const int xido = blockIdx_x * blockDim.x + threadIdx.x;
    const int yido = blockIdx_y * blockDim.y + threadIdx.y;

    const int limages = min(out.dims[2] - setId * nimages, nimages);

    if (xido >= out.dims[0] || yido >= out.dims[1]) return;

    // Compute input index
    typedef typename itype_t<T>::wtype WT;
    WT xidi = xido * t.tmat[0] + yido * t.tmat[1] + t.tmat[2];
    WT yidi = xido * t.tmat[3] + yido * t.tmat[4] + t.tmat[5];

    // Global offset
    //          Offset for transform channel + Offset for image channel.
    int outoff     = setId * nimages * out.strides[2] + batch * out.strides[3];
    int inoff      = setId * nimages * in.strides[2] + batch * in.strides[3];
    const int loco = outoff + (yido * out.strides[1] + xido);

    if (order > 1) {
        // Special conditions to deal with boundaries for bilinear and bicubic
        // FIXME: Ideally this condition should be removed or be present for all
        // methods But tests are expecting a different behavior for bilinear and
        // nearest
        if (xidi < -0.0001 || yidi < -0.0001 || in.dims[0] < xidi ||
            in.dims[1] < yidi) {
            for (int i = 0; i < nimages; i++) {
                out.ptr[loco + i * out.strides[2]] = scalar<T>(0.0f);
            }
            return;
        }
    }

    Interp2<T, WT, 0, 1, order> interp;
    // FIXME: Nearest and lower do not do clamping, but other methods do
    // Make it consistent
    bool clamp = order != 1;
    interp(out, loco, in, inoff, xidi, yidi, method, limages, clamp);
}

}  // namespace cuda
}  // namespace arrayfire
