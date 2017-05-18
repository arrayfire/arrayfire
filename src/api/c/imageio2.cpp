/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(WITH_FREEIMAGE)

#include "imageio_helper.h"

#include <af/array.h>
#include <af/index.h>
#include <af/dim4.hpp>
#include <af/arith.h>
#include <af/algorithm.h>
#include <af/blas.h>
#include <af/data.h>
#include <af/image.h>
#include <backend.hpp>
#include <ArrayInfo.hpp>
#include <traits.hpp>
#include <memory.hpp>
#include <err_common.hpp>
#include <handle.hpp>

#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>

using af::dim4;
using namespace detail;

template<typename T, FI_CHANNELS fi_color>
static af_err readImage_t(af_array *rImage, const uchar* pSrcLine, const int nSrcPitch,
                            const uint fi_w, const uint fi_h)
{
    // create an array to receive the loaded image data.
    AF_CHECK(af_init());
    T *pDst = pinnedAlloc<T>(fi_w * fi_h * 4); // 4 channels is max
    T* pDst0 = pDst;
    T* pDst1 = pDst + (fi_w * fi_h * 1);
    T* pDst2 = pDst + (fi_w * fi_h * 2);
    T* pDst3 = pDst + (fi_w * fi_h * 3);

    uint indx = 0;
    uint step = fi_color;

    for (uint x = 0; x < fi_w; ++x) {
        for (uint y = 0; y < fi_h; ++y) {
            const T *src = (T*)((uchar*)pSrcLine - y * nSrcPitch);
            if(fi_color == 1) {
                pDst0[indx] = (T) *(src + (x * step));
            } else if(fi_color >= 3) {
                if((af_dtype) af::dtype_traits<T>::af_type == u8) {
                    pDst0[indx] = (T) *(src + (x * step + FI_RGBA_RED));
                    pDst1[indx] = (T) *(src + (x * step + FI_RGBA_GREEN));
                    pDst2[indx] = (T) *(src + (x * step + FI_RGBA_BLUE));
                    if (fi_color == 4) pDst3[indx] = (T) *(src + (x * step + FI_RGBA_ALPHA));
                } else {
                    // Non 8-bit types do not use ordering
                    // See Pixel Access Functions Chapter in FreeImage Doc
                    pDst0[indx] = (T) *(src + (x * step + 0));
                    pDst1[indx] = (T) *(src + (x * step + 1));
                    pDst2[indx] = (T) *(src + (x * step + 2));
                    if (fi_color == 4) pDst3[indx] = (T) *(src + (x * step + 3));
                }
            }
            indx++;
        }
    }

    // TODO
    af::dim4 dims(fi_h, fi_w, fi_color, 1);
    af_err err = af_create_array(rImage, pDst, dims.ndims(), dims.get(),
                                 (af_dtype) af::dtype_traits<T>::af_type);
    pinnedFree(pDst);
    return err;
}

FREE_IMAGE_TYPE getFIT(FI_CHANNELS channels, af_dtype type)
{
    if(channels == AFFI_GRAY) {
             if(type == u8 ) return FIT_BITMAP;
        else if(type == u16) return FIT_UINT16;
        else if(type == f32) return FIT_FLOAT;
    } else if(channels == AFFI_RGB) {
             if(type == u8 ) return FIT_BITMAP;
        else if(type == u16) return FIT_RGB16;
        else if(type == f32) return FIT_RGBF;
    } else if(channels == AFFI_RGBA) {
             if(type == u8 ) return FIT_BITMAP;
        else if(type == u16) return FIT_RGBA16;
        else if(type == f32) return FIT_RGBAF;
    }
    return FIT_BITMAP;
}

////////////////////////////////////////////////////////////////////////////////
// File IO
////////////////////////////////////////////////////////////////////////////////
// Load image from disk.
af_err af_load_image_native(af_array *out, const char* filename)
{
    try {
        ARG_ASSERT(1, filename != NULL);

        FI_Init();

        FREE_IMAGE_FORMAT fif = identifyFIF(filename);

        if(fif == FIF_UNKNOWN) {
            AF_ERROR("FreeImage Error: Unknown File or Filetype", AF_ERR_NOT_SUPPORTED);
        }

        int flags = 0;
        if(fif == FIF_JPEG) flags = flags | JPEG_ACCURATE;

        FI_BitmapResource bmp(fif, filename, flags);

        if (bmp.isNotValid()) {
            AF_ERROR("FreeImage Error: Error reading image or file does not exist", AF_ERR_RUNTIME);
        }

        // check image color type
        FREE_IMAGE_COLOR_TYPE colorType = FreeImage_GetColorType(bmp);
        const uint fi_bpp = FreeImage_GetBPP(bmp);

        int fi_color;
        switch(colorType) {
            case FIC_MINISBLACK:
            case FIC_MINISWHITE: fi_color = 1; break;
            case FIC_PALETTE:
            case FIC_RGB:        fi_color = 3; break;
            case FIC_RGBALPHA:
            case FIC_CMYK:       fi_color = 4; break;
            default:             fi_color = 3; break; // Should not come here
        }

        const int fi_bpc = fi_bpp / fi_color;
        if (fi_bpc != 8 && fi_bpc != 16 && fi_bpc != 32) {
            AF_ERROR("FreeImage Error: Bits per channel not supported", AF_ERR_NOT_SUPPORTED);
        }

        // data type
        FREE_IMAGE_TYPE image_type = FreeImage_GetImageType(bmp);

        // sizes
        uint fi_w = FreeImage_GetWidth(bmp);
        uint fi_h = FreeImage_GetHeight(bmp);

        // FI = row major | AF = column major
        uint nSrcPitch = FreeImage_GetPitch(bmp);
        const uchar* pSrcLine = FreeImage_GetBits(bmp) + nSrcPitch * (fi_h - 1);

        // result image
        af_array rImage;
        if(fi_color == 4) {     //4 channel image
            if(fi_bpc == 8)
                AF_CHECK((readImage_t<uchar,  AFFI_RGBA>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
            else if(fi_bpc == 16)
                AF_CHECK((readImage_t<ushort, AFFI_RGBA>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
            else if(fi_bpc == 32)
                switch(image_type) {
                    case FIT_UINT32: AF_CHECK((readImage_t<uint, AFFI_RGBA>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h)); break;
                    case FIT_INT32: AF_CHECK((readImage_t<int, AFFI_RGBA>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h)); break;
                    case FIT_FLOAT: AF_CHECK((readImage_t<float, AFFI_RGBA>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h)); break;
                    default: AF_ERROR("FreeImage Error: Unknown image type", AF_ERR_NOT_SUPPORTED); break;
                }
        } else if (fi_color == 1) {
            if(fi_bpc == 8)
                AF_CHECK((readImage_t<uchar,  AFFI_GRAY>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
            else if(fi_bpc == 16)
                AF_CHECK((readImage_t<ushort, AFFI_GRAY>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
            else if(fi_bpc == 32)
                switch(image_type) {
                    case FIT_UINT32: AF_CHECK((readImage_t<uint,  AFFI_GRAY>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h)); break;
                    case FIT_INT32: AF_CHECK((readImage_t<int,  AFFI_GRAY>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h)); break;
                    case FIT_FLOAT: AF_CHECK((readImage_t<float,  AFFI_GRAY>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h)); break;
                    default: AF_ERROR("FreeImage Error: Unknown image type", AF_ERR_NOT_SUPPORTED); break;
                }
        } else {             //3 channel imag
            if(fi_bpc == 8)
                AF_CHECK((readImage_t<uchar,  AFFI_RGB >)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
            else if(fi_bpc == 16)
                AF_CHECK((readImage_t<ushort, AFFI_RGB >)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
            else if(fi_bpc == 32)
                switch(image_type) {
                    case FIT_UINT32: AF_CHECK((readImage_t<uint,  AFFI_RGB >)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h)); break;
                    case FIT_INT32: AF_CHECK((readImage_t<int,  AFFI_RGB >)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h)); break;
                    case FIT_FLOAT: AF_CHECK((readImage_t<float,  AFFI_RGB >)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h)); break;
                    default: AF_ERROR("FreeImage Error: Unknown image type", AF_ERR_NOT_SUPPORTED); break;
                }
        }

        std::swap(*out,rImage);
    } CATCHALL;

    return AF_SUCCESS;
}

template<typename T, FI_CHANNELS channels>
static void save_t(T* pDstLine, const af_array in, const dim4 dims, uint nDstPitch)
{
    af_array rr = 0, gg = 0, bb = 0, aa = 0;
    AF_CHECK(channel_split(in, dims, &rr, &gg, &bb, &aa)); // convert array to 3 channels if needed

    af_array rrT = 0, ggT = 0, bbT = 0, aaT = 0;
    T *pSrc0 = 0, *pSrc1 = 0, *pSrc2 = 0, *pSrc3 = 0;

    uint step = channels; // force 3 channels saving
    uint indx = 0;

                      AF_CHECK(af_transpose(&rrT, rr, false));
    if(channels >= 3) AF_CHECK(af_transpose(&ggT, gg, false));
    if(channels >= 3) AF_CHECK(af_transpose(&bbT, bb, false));
    if(channels >= 4) AF_CHECK(af_transpose(&aaT, aa, false));

    const ArrayInfo& cinfo = getInfo(rrT);
                      pSrc0 = pinnedAlloc<T>(cinfo.elements());
    if(channels >= 3) pSrc1 = pinnedAlloc<T>(cinfo.elements());
    if(channels >= 3) pSrc2 = pinnedAlloc<T>(cinfo.elements());
    if(channels >= 4) pSrc3 = pinnedAlloc<T>(cinfo.elements());

                      AF_CHECK(af_get_data_ptr((void*)pSrc0, rrT));
    if(channels >= 3) AF_CHECK(af_get_data_ptr((void*)pSrc1, ggT));
    if(channels >= 3) AF_CHECK(af_get_data_ptr((void*)pSrc2, bbT));
    if(channels >= 4) AF_CHECK(af_get_data_ptr((void*)pSrc3, aaT));

    const uint fi_w = dims[1];
    const uint fi_h = dims[0];

    // Copy the array into FreeImage buffer
    for (uint y = 0; y < fi_h; ++y) {
        for (uint x = 0; x < fi_w; ++x) {
            if(channels == 1) {
                *(pDstLine + x * step) = (T) pSrc0[indx]; // r -> 0
            } else if(channels >=3) {
                if((af_dtype) af::dtype_traits<T>::af_type == u8) {
                    *(pDstLine + x * step + FI_RGBA_RED  ) = (T) pSrc0[indx]; // r -> 0
                    *(pDstLine + x * step + FI_RGBA_GREEN) = (T) pSrc1[indx]; // g -> 1
                    *(pDstLine + x * step + FI_RGBA_BLUE ) = (T) pSrc2[indx]; // b -> 2
                    if(channels >= 4) *(pDstLine + x * step + FI_RGBA_ALPHA) = (T) pSrc3[indx]; // a
                } else {
                    // Non 8-bit types do not use ordering
                    // See Pixel Access Functions Chapter in FreeImage Doc
                    *(pDstLine + x * step + 0) = (T) pSrc0[indx]; // r -> 0
                    *(pDstLine + x * step + 1) = (T) pSrc1[indx]; // g -> 1
                    *(pDstLine + x * step + 2) = (T) pSrc2[indx]; // b -> 2
                    if(channels >= 4) *(pDstLine + x * step + 3) = (T) pSrc3[indx]; // a
                }
            }
            ++indx;
        }
        pDstLine = (T*)(((uchar*)pDstLine) - nDstPitch);
    }
                      pinnedFree(pSrc0);
    if(channels >= 3) pinnedFree(pSrc1);
    if(channels >= 3) pinnedFree(pSrc2);
    if(channels >= 4) pinnedFree(pSrc3);

    if(rr != 0) AF_CHECK(af_release_array(rr ));
    if(gg != 0) AF_CHECK(af_release_array(gg ));
    if(bb != 0) AF_CHECK(af_release_array(bb ));
    if(aa != 0) AF_CHECK(af_release_array(aa ));
    if(rrT!= 0) AF_CHECK(af_release_array(rrT));
    if(ggT!= 0) AF_CHECK(af_release_array(ggT));
    if(bbT!= 0) AF_CHECK(af_release_array(bbT));
    if(aaT!= 0) AF_CHECK(af_release_array(aaT));
}

// Save an image to disk.
af_err af_save_image_native(const char* filename, const af_array in)
{
    try {

        ARG_ASSERT(0, filename != NULL);

        FI_Init();

        FREE_IMAGE_FORMAT fif = identifyFIF(filename);

        if(fif == FIF_UNKNOWN) {
            AF_ERROR("FreeImage Error: Unknown Filetype", AF_ERR_NOT_SUPPORTED);
        }

        const ArrayInfo& info = getInfo(in);
        // check image color type
        FI_CHANNELS channels = (FI_CHANNELS)info.dims()[2];
        DIM_ASSERT(1, channels <= 4);
        DIM_ASSERT(1, channels != 2);

        // sizes
        uint fi_w = info.dims()[1];
        uint fi_h = info.dims()[0];

        af_dtype type = info.getType();

        // FI assumes [0-255] for u8
        // FI assumes [0-65k] for u16
        // FI assumes [0-1]   for f32
        int fi_bpp = 0;
        switch(type) {
            case u8:  fi_bpp = channels * 8; break;
            case u16: fi_bpp = channels * 16; break;
            case f32: fi_bpp = channels * 32; break;
            default: TYPE_ERROR(1, type);
        }

        FREE_IMAGE_TYPE fit_type = getFIT(channels, type);

        FI_BitmapResource bmp(fi_w, fi_h, fi_bpp, fit_type);

        if (bmp.isNotValid()) {
            AF_ERROR("FreeImage Error: Error creating image or file", AF_ERR_RUNTIME);
        }

        // FI = row major | AF = column major
        uint nDstPitch = FreeImage_GetPitch(bmp);
        void* pDstLine = FreeImage_GetBits(bmp) + nDstPitch * (fi_h - 1);

        if(channels == AFFI_GRAY) {
            switch(type) {
                case u8:  save_t<uchar , AFFI_GRAY>((uchar *)pDstLine, in, info.dims(), nDstPitch); break;
                case u16: save_t<ushort, AFFI_GRAY>((ushort*)pDstLine, in, info.dims(), nDstPitch); break;
                case f32: save_t<float , AFFI_GRAY>((float *)pDstLine, in, info.dims(), nDstPitch); break;
                default: TYPE_ERROR(1, type);
            }
        } else if(channels == AFFI_RGB) {
            switch(type) {
                case u8:  save_t<uchar , AFFI_RGB >((uchar *)pDstLine, in, info.dims(), nDstPitch); break;
                case u16: save_t<ushort, AFFI_RGB >((ushort*)pDstLine, in, info.dims(), nDstPitch); break;
                case f32: save_t<float , AFFI_RGB >((float *)pDstLine, in, info.dims(), nDstPitch); break;
                default: TYPE_ERROR(1, type);
            }
        } else {
            switch(type) {
                case u8:  save_t<uchar , AFFI_RGBA>((uchar *)pDstLine, in, info.dims(), nDstPitch); break;
                case u16: save_t<ushort, AFFI_RGBA>((ushort*)pDstLine, in, info.dims(), nDstPitch); break;
                case f32: save_t<float , AFFI_RGBA>((float *)pDstLine, in, info.dims(), nDstPitch); break;
                default: TYPE_ERROR(1, type);
            }
        }

        int flags = 0;
        if(fif == FIF_JPEG) flags = flags | JPEG_QUALITYSUPERB;

        // now save the result image
        if (!(FreeImage_Save(fif, bmp, filename, flags) == TRUE)) {
            AF_ERROR("FreeImage Error: Failed to save image", AF_ERR_RUNTIME);
        }

    } CATCHALL

    return AF_SUCCESS;
}

af_err af_is_image_io_available(bool *out)
{
    *out = true;
    return AF_SUCCESS;
}

#else   // WITH_FREEIMAGE
#include <af/image.h>
#include <stdio.h>
#include <err_common.hpp>
af_err af_load_image_native(af_array *out, const char* filename)
{
    AF_RETURN_ERROR("ArrayFire compiled without Image IO (FreeImage) support", AF_ERR_NOT_CONFIGURED);
}

af_err af_save_image_native(const char* filename, const af_array in)
{
    AF_RETURN_ERROR("ArrayFire compiled without Image IO (FreeImage) support", AF_ERR_NOT_CONFIGURED);
}

af_err af_is_image_io_available(bool *out)
{
    *out = false;
    return AF_SUCCESS;
}
#endif  // WITH_FREEIMAGE
