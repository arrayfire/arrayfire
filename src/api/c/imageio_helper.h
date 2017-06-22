/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifndef IMAGEIO_HELPER_H
#define IMAGEIO_HELPER_H

#include <FreeImage.h>

#include <af/array.h>
#include <af/index.h>
#include <af/dim4.hpp>
#include <common/err_common.hpp>

class FI_Manager
{
    public:
    FI_Manager()
    {
#ifdef FREEIMAGE_LIB
        FreeImage_Initialise();
#endif
    }

    ~FI_Manager()
    {
#ifdef FREEIMAGE_LIB
        FreeImage_DeInitialise();
#endif
    }
};

static void FI_Init()
{
    static FI_Manager manager = FI_Manager();
}

class FI_BitmapResource
{
public:
    explicit FI_BitmapResource(FIBITMAP * p) :
        pBitmap(p)
    {
    }

    ~FI_BitmapResource()
    {
        FreeImage_Unload(pBitmap);
    }
private:
    FIBITMAP * pBitmap;
};

typedef enum {
    AFFI_GRAY = 1,
    AFFI_RGB  = 3,
    AFFI_RGBA = 4
} FI_CHANNELS;

// Error handler for FreeImage library.
// In case this handler is invoked, it throws an af exception.
static void FreeImageErrorHandler(FREE_IMAGE_FORMAT oFif, const char* zMessage)
{
    printf("FreeImage Error Handler: %s\n", zMessage);
}

//  Split a MxNx3 image into 3 separate channel matrices.
//  Produce 3 channels if needed
static af_err channel_split(const af_array rgb, const af::dim4 &dims,
                            af_array *outr, af_array *outg, af_array *outb, af_array *outa)
{
    try {
        af_seq idx[4][3] = {{af_span, af_span, {0, 0, 1}},
                            {af_span, af_span, {1, 1, 1}},
                            {af_span, af_span, {2, 2, 1}},
                            {af_span, af_span, {3, 3, 1}}
                           };

        if (dims[2] == 4) {
            AF_CHECK(af_index(outr, rgb, dims.ndims(), idx[0]));
            AF_CHECK(af_index(outg, rgb, dims.ndims(), idx[1]));
            AF_CHECK(af_index(outb, rgb, dims.ndims(), idx[2]));
            AF_CHECK(af_index(outa, rgb, dims.ndims(), idx[3]));
        } else if (dims[2] == 3) {
            AF_CHECK(af_index(outr, rgb, dims.ndims(), idx[0]));
            AF_CHECK(af_index(outg, rgb, dims.ndims(), idx[1]));
            AF_CHECK(af_index(outb, rgb, dims.ndims(), idx[2]));
        } else {
            AF_CHECK(af_index(outr, rgb, dims.ndims(), idx[0]));
        }
    } CATCHALL;
    return AF_SUCCESS;
}

#endif
