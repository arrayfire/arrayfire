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
#include <err_common.hpp>

// Error handler for FreeImage library.
// In case this handler is invoked, it throws an af exception.
static void FreeImageErrorHandler(FREE_IMAGE_FORMAT oFif, const char* zMessage)
{
    printf("FreeImage Error Handler: %s\n", zMessage);
}

class FI_Manager
{
    public:
    static bool initialized;
    FI_Manager()
    {
#ifdef FREEIMAGE_LIB
        FreeImage_Initialise();
#endif
        initialized = true;
        // set your own FreeImage error handler
        FreeImage_SetOutputMessage(FreeImageErrorHandler);
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
    private:
        FIBITMAP * mBitmap;

    public:
        FI_BitmapResource(FREE_IMAGE_FORMAT fif, const char* filename, int flags)
        {
            // check that the plugin has reading capabilities ...
            if (FreeImage_FIFSupportsReading(fif)) {
                mBitmap = FreeImage_Load(fif, filename, flags);
            }
        }

        FI_BitmapResource(FREE_IMAGE_FORMAT fif, FIMEMORY* stream, int flags)
        {
            if (FreeImage_FIFSupportsReading(fif)) {
                mBitmap = FreeImage_LoadFromMemory(fif, stream, flags);
            }
        }

        FI_BitmapResource(unsigned w, unsigned h, unsigned bpp, FREE_IMAGE_TYPE fitType=FIT_BITMAP)
            : mBitmap(FreeImage_AllocateT(fitType, w, h, bpp))
        {
        }

        ~FI_BitmapResource()
        {
            FreeImage_Unload(mBitmap);
        }

        bool isNotValid() const { return mBitmap==NULL; }

        operator FIBITMAP* () { return mBitmap; }
};

static inline
FREE_IMAGE_FORMAT identifyFIF(const char* filename)
{
    FREE_IMAGE_FORMAT retVal = FreeImage_GetFileType(filename);

    if (retVal == FIF_UNKNOWN) {
        retVal = FreeImage_GetFIFFromFilename(filename);
    }

    return retVal;
}

typedef enum {
    AFFI_GRAY = 1,
    AFFI_RGB  = 3,
    AFFI_RGBA = 4
} FI_CHANNELS;

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
