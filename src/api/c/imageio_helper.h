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

#include <common/DependencyModule.hpp>
#include <common/err_common.hpp>
#include <af/array.h>
#include <af/dim4.hpp>
#include <af/index.h>

#include <FreeImage.h>

#include <functional>
#include <memory>

namespace arrayfire {

class FreeImage_Module {
    common::DependencyModule module;

   public:
    MODULE_MEMBER(FreeImage_Allocate);
    MODULE_MEMBER(FreeImage_AllocateT);
    MODULE_MEMBER(FreeImage_CloseMemory);
    MODULE_MEMBER(FreeImage_DeInitialise);
    MODULE_MEMBER(FreeImage_FIFSupportsReading);
    MODULE_MEMBER(FreeImage_GetBPP);
    MODULE_MEMBER(FreeImage_GetBits);
    MODULE_MEMBER(FreeImage_GetColorType);
    MODULE_MEMBER(FreeImage_GetFIFFromFilename);
    MODULE_MEMBER(FreeImage_GetFileType);
    MODULE_MEMBER(FreeImage_GetFileTypeFromMemory);
    MODULE_MEMBER(FreeImage_GetHeight);
    MODULE_MEMBER(FreeImage_GetImageType);
    MODULE_MEMBER(FreeImage_GetPitch);
    MODULE_MEMBER(FreeImage_GetWidth);
    MODULE_MEMBER(FreeImage_Initialise);
    MODULE_MEMBER(FreeImage_Load);
    MODULE_MEMBER(FreeImage_LoadFromMemory);
    MODULE_MEMBER(FreeImage_OpenMemory);
    MODULE_MEMBER(FreeImage_Save);
    MODULE_MEMBER(FreeImage_SaveToMemory);
    MODULE_MEMBER(FreeImage_SeekMemory);
    MODULE_MEMBER(FreeImage_SetOutputMessage);
    MODULE_MEMBER(FreeImage_Unload);

    FreeImage_Module();
    ~FreeImage_Module();
};

FreeImage_Module &getFreeImagePlugin();

using bitmap_ptr = std::unique_ptr<FIBITMAP, std::function<void(FIBITMAP *)>>;
bitmap_ptr make_bitmap_ptr(FIBITMAP *);

typedef enum {
    AFFI_GRAY = 1,  //< gray
    AFFI_RGB  = 3,  //< rgb
    AFFI_RGBA = 4   //< rgba
} FI_CHANNELS;

// Error handler for FreeImage library.
// In case this handler is invoked, it throws an af exception.
static void FreeImageErrorHandler(FREE_IMAGE_FORMAT oFif,
                                  const char *zMessage) {
    UNUSED(oFif);
    printf("FreeImage Error Handler: %s\n", zMessage);
}

//  Split a MxNx3 image into 3 separate channel matrices.
//  Produce 3 channels if needed
static af_err channel_split(const af_array rgb, const af::dim4 &dims,
                            af_array *outr, af_array *outg, af_array *outb,
                            af_array *outa) {
    try {
        af_seq idx[4][3] = {{af_span, af_span, {0, 0, 1}},
                            {af_span, af_span, {1, 1, 1}},
                            {af_span, af_span, {2, 2, 1}},
                            {af_span, af_span, {3, 3, 1}}};

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
    }
    CATCHALL;
    return AF_SUCCESS;
}

#endif
}
