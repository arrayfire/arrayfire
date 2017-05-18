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

bool FI_Manager::initialized = false;

template<typename T, FI_CHANNELS fi_color, FI_CHANNELS fo_color>
static af_err readImage(af_array *rImage, const uchar* pSrcLine, const int nSrcPitch,
                        const uint fi_w, const uint fi_h)
{
    // create an array to receive the loaded image data.
    AF_CHECK(af_init());
    float *pDst = pinnedAlloc<float>(fi_w * fi_h * 4); // 4 channels is max
    float* pDst0 = pDst;
    float* pDst1 = pDst + (fi_w * fi_h * 1);
    float* pDst2 = pDst + (fi_w * fi_h * 2);
    float* pDst3 = pDst + (fi_w * fi_h * 3);

    uint indx = 0;
    uint step = fi_color;

    for (uint x = 0; x < fi_w; ++x) {
        for (uint y = 0; y < fi_h; ++y) {
            const T *src = (T*)(pSrcLine - y * nSrcPitch);
            if(fo_color == 1) {
                pDst0[indx] = (T) *(src + (x * step));
            } else if(fo_color >= 3) {
                if((af_dtype) af::dtype_traits<T>::af_type == u8) {
                    pDst0[indx] = (float) *(src + (x * step + FI_RGBA_RED));
                    pDst1[indx] = (float) *(src + (x * step + FI_RGBA_GREEN));
                    pDst2[indx] = (float) *(src + (x * step + FI_RGBA_BLUE));
                    if (fo_color == 4) pDst3[indx] = (float) *(src + (x * step + FI_RGBA_ALPHA));
                } else {
                    // Non 8-bit types do not use ordering
                    // See Pixel Access Functions Chapter in FreeImage Doc
                    pDst0[indx] = (float) *(src + (x * step + 0));
                    pDst1[indx] = (float) *(src + (x * step + 1));
                    pDst2[indx] = (float) *(src + (x * step + 2));
                    if (fo_color == 4) pDst3[indx] = (float) *(src + (x * step + 3));
                }
            }
            indx++;
        }
    }

    // TODO
    af::dim4 dims(fi_h, fi_w, fo_color, 1);
    af_err err = af_create_array(rImage, pDst, dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<float>::af_type);
    pinnedFree(pDst);
    return err;
}

template<typename T, FI_CHANNELS fo_color>
static af_err readImage(af_array *rImage, const uchar* pSrcLine, const int nSrcPitch,
                        const uint fi_w, const uint fi_h)
{
    // create an array to receive the loaded image data.
    AF_CHECK(af_init());
    float *pDst = pinnedAlloc<float>(fi_w * fi_h);

    uint indx = 0;
    uint step = nSrcPitch / (fi_w * sizeof(T));
    T r, g, b;
    for (uint x = 0; x < fi_w; ++x) {
        for (uint y = 0; y < fi_h; ++y) {
            const T *src = (T*)(pSrcLine - y * nSrcPitch);
            if(fo_color == 1) {
                pDst[indx] = (T) *(src + (x * step));
            } else if(fo_color >= 3) {
                if((af_dtype) af::dtype_traits<T>::af_type == u8) {
                    r = (T) *(src + (x * step + FI_RGBA_RED));
                    g = (T) *(src + (x * step + FI_RGBA_GREEN));
                    b = (T) *(src + (x * step + FI_RGBA_BLUE));
                } else {
                    // Non 8-bit types do not use ordering
                    // See Pixel Access Functions Chapter in FreeImage Doc
                    r = (T) *(src + (x * step + 0));
                    g = (T) *(src + (x * step + 1));
                    b = (T) *(src + (x * step + 2));
                }
                pDst[indx] = r * 0.2989f + g * 0.5870f + b * 0.1140f;
            }
            indx++;
        }
    }

    af::dim4 dims(fi_h, fi_w, 1, 1);
    af_err err = af_create_array(rImage, pDst, dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<float>::af_type);
    pinnedFree(pDst);
    return err;
}

////////////////////////////////////////////////////////////////////////////////
// File IO
////////////////////////////////////////////////////////////////////////////////
// Load image from disk.
af_err af_load_image(af_array *out, const char* filename, const bool isColor)
{
    try {
        ARG_ASSERT(1, filename != NULL);

        // for statically linked FI
        FI_Init();

        // try to guess the file format from the file extension
        FREE_IMAGE_FORMAT fif = identifyFIF(filename);

        if(fif == FIF_UNKNOWN) {
            AF_ERROR("FreeImage Error: Unknown File or Filetype", AF_ERR_NOT_SUPPORTED);
        }

        int flags = 0;
        if(fif == FIF_JPEG) flags = flags | JPEG_ACCURATE;
#ifdef JPEG_GREYSCALE
        if(fif == FIF_JPEG && !isColor) flags = flags | JPEG_GREYSCALE;
#endif

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
        if(fi_bpc != 8 && fi_bpc != 16 && fi_bpc != 32) {
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
        if (isColor) {
            if(fi_color == 4) {     //4 channel image
                if(fi_bpc == 8)
                    AF_CHECK((readImage<uchar,  AFFI_RGBA, AFFI_RGBA>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
                else if(fi_bpc == 16)
                    AF_CHECK((readImage<ushort, AFFI_RGBA, AFFI_RGBA>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
                else if(fi_bpc == 32)
                    switch(image_type) {
                        case FIT_UINT32: AF_CHECK((readImage<uint,  AFFI_RGBA, AFFI_RGBA>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h)); break;
                        case FIT_INT32: AF_CHECK((readImage<int,   AFFI_RGBA, AFFI_RGBA>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h)); break;
                        case FIT_FLOAT: AF_CHECK((readImage<float,  AFFI_RGBA, AFFI_RGBA>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h)); break;
                        default: AF_ERROR("FreeImage Error: Unknown image type", AF_ERR_NOT_SUPPORTED); break;
                    }
            } else if (fi_color == 1) {
                if(fi_bpc == 8)
                    AF_CHECK((readImage<uchar,  AFFI_GRAY, AFFI_RGB>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
                else if(fi_bpc == 16)
                    AF_CHECK((readImage<ushort, AFFI_GRAY, AFFI_RGB>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
                else if(fi_bpc == 32)
                    switch(image_type) {
                        case FIT_UINT32: AF_CHECK((readImage<uint,  AFFI_GRAY, AFFI_RGB>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h)); break;
                        case FIT_INT32: AF_CHECK((readImage<int,   AFFI_GRAY, AFFI_RGB>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h)); break;
                        case FIT_FLOAT: AF_CHECK((readImage<float,  AFFI_GRAY, AFFI_RGB>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h)); break;
                        default: AF_ERROR("FreeImage Error: Unknown image type", AF_ERR_NOT_SUPPORTED); break;
                    }
            } else {             //3 channel image
                if(fi_bpc == 8)
                    AF_CHECK((readImage<uchar,  AFFI_RGB, AFFI_RGB>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
                else if(fi_bpc == 16)
                    AF_CHECK((readImage<ushort, AFFI_RGB, AFFI_RGB>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
                else if(fi_bpc == 32)
                    switch(image_type) {
                        case FIT_UINT32: AF_CHECK((readImage<uint,  AFFI_RGB, AFFI_RGB>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h)); break;
                        case FIT_INT32: AF_CHECK((readImage<int,   AFFI_RGB, AFFI_RGB>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h)); break;
                        case FIT_FLOAT: AF_CHECK((readImage<float,  AFFI_RGB, AFFI_RGB>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h)); break;
                        default: AF_ERROR("FreeImage Error: Unknown image type", AF_ERR_NOT_SUPPORTED); break;
                    }
            }
        } else {                    //output gray irrespective
            if(fi_color == 1) {     //4 channel image
                if(fi_bpc == 8)
                    AF_CHECK((readImage<uchar,  AFFI_GRAY>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
                else if(fi_bpc == 16)
                    AF_CHECK((readImage<ushort, AFFI_GRAY>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
                else if(fi_bpc == 32)
                    switch(image_type) {
                        case FIT_UINT32: AF_CHECK((readImage<uint,  AFFI_GRAY>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h)); break;
                        case FIT_INT32: AF_CHECK((readImage<int,   AFFI_GRAY>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h)); break;
                        case FIT_FLOAT: AF_CHECK((readImage<float,  AFFI_GRAY>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h)); break;
                        default: AF_ERROR("FreeImage Error: Unknown image type", AF_ERR_NOT_SUPPORTED); break;
                    }
            } else if (fi_color == 3 || fi_color == 4) {
                if(fi_bpc == 8)
                    AF_CHECK((readImage<uchar,  AFFI_RGB>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
                else if(fi_bpc == 16)
                    AF_CHECK((readImage<ushort, AFFI_RGB>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
                else if(fi_bpc == 32)
                    switch(image_type) {
                        case FIT_UINT32: AF_CHECK((readImage<uint,  AFFI_RGB>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h)); break;
                        case FIT_INT32: AF_CHECK((readImage<int,   AFFI_RGB>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h)); break;
                        case FIT_FLOAT: AF_CHECK((readImage<float,  AFFI_RGB>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h)); break;
                        default: AF_ERROR("FreeImage Error: Unknown image type", AF_ERR_NOT_SUPPORTED); break;
                    }
            }
        }

        std::swap(*out,rImage);
    } CATCHALL;

    return AF_SUCCESS;
}

// Save an image to disk.
af_err af_save_image(const char* filename, const af_array in_)
{
    try {

        ARG_ASSERT(0, filename != NULL);

        FI_Init();

        // try to guess the file format from the file extension
        FREE_IMAGE_FORMAT fif = identifyFIF(filename);

        if(fif == FIF_UNKNOWN) {
            AF_ERROR("FreeImage Error: Unknown Filetype", AF_ERR_NOT_SUPPORTED);
        }

        const ArrayInfo& info = getInfo(in_);
        // check image color type
        uint channels = info.dims()[2];
        DIM_ASSERT(1, channels <= 4);
        DIM_ASSERT(1, channels != 2);

        int fi_bpp = channels * 8;

        // sizes
        uint fi_w = info.dims()[1];
        uint fi_h = info.dims()[0];

        FI_BitmapResource bmp(fi_w, fi_h, fi_bpp);

        if (bmp.isNotValid()) {
            AF_ERROR("FreeImage Error: Error creating image or file", AF_ERR_RUNTIME);
        }

        // FI assumes [0-255]
        // If array is in 0-1 range, multiply by 255
        af_array in;
        double max_real, max_imag;
        bool free_in = false;
        AF_CHECK(af_max_all(&max_real, &max_imag, in_));
        if (max_real <= 1) {
            af_array c255 = 0;
            AF_CHECK(af_constant(&c255, 255.0, info.ndims(), info.dims().get(), f32));
            AF_CHECK(af_mul(&in, in_, c255, false));
            AF_CHECK(af_release_array(c255));
            free_in = true;
        } else if(max_real < 256) {
            in = in_;
        } else if (max_real < 65536) {
            af_array c255 = 0;
            AF_CHECK(af_constant(&c255, 257.0, info.ndims(), info.dims().get(), f32));
            AF_CHECK(af_div(&in, in_, c255, false));
            AF_CHECK(af_release_array(c255));
            free_in = true;
        } else {
            in = in_;
        }

        // FI = row major | AF = column major
        uint nDstPitch = FreeImage_GetPitch(bmp);
        uchar* pDstLine = FreeImage_GetBits(bmp) + nDstPitch * (fi_h - 1);
        af_array rr = 0, gg = 0, bb = 0, aa = 0;
        AF_CHECK(channel_split(in, info.dims(), &rr, &gg, &bb, &aa)); // convert array to 3 channels if needed

        uint step = channels; // force 3 channels saving
        uint indx = 0;

        af_array rrT = 0, ggT = 0, bbT = 0, aaT = 0;
        if(channels == 4) {

            AF_CHECK(af_transpose(&rrT, rr, false));
            AF_CHECK(af_transpose(&ggT, gg, false));
            AF_CHECK(af_transpose(&bbT, bb, false));
            AF_CHECK(af_transpose(&aaT, aa, false));

            const ArrayInfo& cinfo = getInfo(rrT);
            float* pSrc0 = pinnedAlloc<float>(cinfo.elements());
            float* pSrc1 = pinnedAlloc<float>(cinfo.elements());
            float* pSrc2 = pinnedAlloc<float>(cinfo.elements());
            float* pSrc3 = pinnedAlloc<float>(cinfo.elements());

            AF_CHECK(af_get_data_ptr((void*)pSrc0, rrT));
            AF_CHECK(af_get_data_ptr((void*)pSrc1, ggT));
            AF_CHECK(af_get_data_ptr((void*)pSrc2, bbT));
            AF_CHECK(af_get_data_ptr((void*)pSrc3, aaT));

            // Copy the array into FreeImage buffer
            for (uint y = 0; y < fi_h; ++y) {
                for (uint x = 0; x < fi_w; ++x) {
                    *(pDstLine + x * step + FI_RGBA_RED  ) = (uchar) pSrc0[indx]; // r
                    *(pDstLine + x * step + FI_RGBA_GREEN) = (uchar) pSrc1[indx]; // g
                    *(pDstLine + x * step + FI_RGBA_BLUE ) = (uchar) pSrc2[indx]; // b
                    *(pDstLine + x * step + FI_RGBA_ALPHA) = (uchar) pSrc3[indx]; // a
                    ++indx;
                }
                pDstLine -= nDstPitch;
            }
            pinnedFree(pSrc0);
            pinnedFree(pSrc1);
            pinnedFree(pSrc2);
            pinnedFree(pSrc3);
        } else if(channels == 3) {
            AF_CHECK(af_transpose(&rrT, rr, false));
            AF_CHECK(af_transpose(&ggT, gg, false));
            AF_CHECK(af_transpose(&bbT, bb, false));

            const ArrayInfo& cinfo = getInfo(rrT);
            float* pSrc0 = pinnedAlloc<float>(cinfo.elements());
            float* pSrc1 = pinnedAlloc<float>(cinfo.elements());
            float* pSrc2 = pinnedAlloc<float>(cinfo.elements());

            AF_CHECK(af_get_data_ptr((void*)pSrc0, rrT));
            AF_CHECK(af_get_data_ptr((void*)pSrc1, ggT));
            AF_CHECK(af_get_data_ptr((void*)pSrc2, bbT));

            // Copy the array into FreeImage buffer
            for (uint y = 0; y < fi_h; ++y) {
                for (uint x = 0; x < fi_w; ++x) {
                    *(pDstLine + x * step + FI_RGBA_RED  ) = (uchar) pSrc0[indx]; // r
                    *(pDstLine + x * step + FI_RGBA_GREEN) = (uchar) pSrc1[indx]; // g
                    *(pDstLine + x * step + FI_RGBA_BLUE ) = (uchar) pSrc2[indx]; // b
                    ++indx;
                }
                pDstLine -= nDstPitch;
            }
            pinnedFree(pSrc0);
            pinnedFree(pSrc1);
            pinnedFree(pSrc2);
        } else {
            AF_CHECK(af_transpose(&rrT, rr, false));
            const ArrayInfo& cinfo = getInfo(rrT);
            float* pSrc0 = pinnedAlloc<float>(cinfo.elements());
            AF_CHECK(af_get_data_ptr((void*)pSrc0, rrT));

            for (uint y = 0; y < fi_h; ++y) {
                for (uint x = 0; x < fi_w; ++x) {
                    *(pDstLine + x * step) = (uchar) pSrc0[indx];
                    ++indx;
                }
                pDstLine -= nDstPitch;
            }
            pinnedFree(pSrc0);
        }

        int flags = 0;
        if(fif == FIF_JPEG) flags = flags | JPEG_QUALITYSUPERB;

        // now save the result image
        if (!(FreeImage_Save(fif, bmp, filename, flags) == TRUE)) {
            AF_ERROR("FreeImage Error: Failed to save image", AF_ERR_RUNTIME);
        }

        if(free_in) AF_CHECK(af_release_array(in ));
        if(rr != 0) AF_CHECK(af_release_array(rr ));
        if(gg != 0) AF_CHECK(af_release_array(gg ));
        if(bb != 0) AF_CHECK(af_release_array(bb ));
        if(aa != 0) AF_CHECK(af_release_array(aa ));
        if(rrT!= 0) AF_CHECK(af_release_array(rrT));
        if(ggT!= 0) AF_CHECK(af_release_array(ggT));
        if(bbT!= 0) AF_CHECK(af_release_array(bbT));
        if(aaT!= 0) AF_CHECK(af_release_array(aaT));

    } CATCHALL

    return AF_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
// Memory IO
////////////////////////////////////////////////////////////////////////////////
/// Load image from memory.
af_err af_load_image_memory(af_array *out, const void* ptr)
{
    try {
        ARG_ASSERT(1, ptr != NULL);

        // for statically linked FI
        FI_Init();

        FIMEMORY *stream = (FIMEMORY*)ptr;
        FreeImage_SeekMemory(stream, 0L, SEEK_SET);

        // try to guess the file format from the file extension
        FREE_IMAGE_FORMAT fif = FreeImage_GetFileTypeFromMemory(stream, 0);

        if(fif == FIF_UNKNOWN) {
            AF_ERROR("FreeImage Error: Unknown File or Filetype", AF_ERR_NOT_SUPPORTED);
        }

        int flags = 0;
        if(fif == FIF_JPEG) flags = flags | JPEG_ACCURATE;

        FI_BitmapResource bmp(fif, stream, flags);

        if(bmp.isNotValid()) {
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
        if(fi_bpc != 8 && fi_bpc != 16 && fi_bpc != 32) {
            AF_ERROR("FreeImage Error: Bits per channel not supported", AF_ERR_NOT_SUPPORTED);
        }

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
                AF_CHECK((readImage<uchar,  AFFI_RGBA, AFFI_RGBA>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
            else if(fi_bpc == 16)
                AF_CHECK((readImage<ushort, AFFI_RGBA, AFFI_RGBA>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
            else if(fi_bpc == 32)
                AF_CHECK((readImage<float,  AFFI_RGBA, AFFI_RGBA>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
        } else if (fi_color == 1) { // 1 channel image
            if(fi_bpc == 8)
                AF_CHECK((readImage<uchar,  AFFI_GRAY>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
            else if(fi_bpc == 16)
                AF_CHECK((readImage<ushort, AFFI_GRAY>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
            else if(fi_bpc == 32)
                AF_CHECK((readImage<float,  AFFI_GRAY>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
        } else {             //3 channel image
            if(fi_bpc == 8)
                AF_CHECK((readImage<uchar,  AFFI_RGB, AFFI_RGB>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
            else if(fi_bpc == 16)
                AF_CHECK((readImage<ushort, AFFI_RGB, AFFI_RGB>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
            else if(fi_bpc == 32)
                AF_CHECK((readImage<float,  AFFI_RGB, AFFI_RGB>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
        }

        std::swap(*out,rImage);
    } CATCHALL;

    return AF_SUCCESS;
}

// Save an image to memory.
af_err af_save_image_memory(void **ptr, const af_array in_, const af_image_format format)
{
    try {

        FI_Init();

        // try to guess the file format from the file extension
        FREE_IMAGE_FORMAT fif = (FREE_IMAGE_FORMAT)format;

        if(fif == FIF_UNKNOWN || fif > 34) { // FreeImage FREE_IMAGE_FORMAT has upto 34 enums as of 3.17
            AF_ERROR("FreeImage Error: Unknown Filetype", AF_ERR_NOT_SUPPORTED);
        }

        const ArrayInfo& info = getInfo(in_);
        // check image color type
        uint channels = info.dims()[2];
        DIM_ASSERT(1, channels <= 4);
        DIM_ASSERT(1, channels != 2);

        int fi_bpp = channels * 8;

        // sizes
        uint fi_w = info.dims()[1];
        uint fi_h = info.dims()[0];

        FI_BitmapResource bmp(fi_w, fi_h, fi_bpp);

        if (bmp.isNotValid()) {
            AF_ERROR("FreeImage Error: Error creating image or file", AF_ERR_RUNTIME);
        }

        // FI assumes [0-255]
        // If array is in 0-1 range, multiply by 255
        af_array in;
        double max_real, max_imag;
        bool free_in = false;
        AF_CHECK(af_max_all(&max_real, &max_imag, in_));
        if (max_real <= 1) {
            af_array c255;
            AF_CHECK(af_constant(&c255, 255.0, info.ndims(), info.dims().get(), f32));
            AF_CHECK(af_mul(&in, in_, c255, false));
            AF_CHECK(af_release_array(c255));
            free_in = true;
        } else {
            in = in_;
        }

        // FI = row major | AF = column major
        uint nDstPitch = FreeImage_GetPitch(bmp);
        uchar* pDstLine = FreeImage_GetBits(bmp) + nDstPitch * (fi_h - 1);
        af_array rr = 0, gg = 0, bb = 0, aa = 0;
        AF_CHECK(channel_split(in, info.dims(), &rr, &gg, &bb, &aa)); // convert array to 3 channels if needed

        uint step = channels; // force 3 channels saving
        uint indx = 0;

        af_array rrT = 0, ggT = 0, bbT = 0, aaT = 0;
        if(channels == 4) {

            AF_CHECK(af_transpose(&rrT, rr, false));
            AF_CHECK(af_transpose(&ggT, gg, false));
            AF_CHECK(af_transpose(&bbT, bb, false));
            AF_CHECK(af_transpose(&aaT, aa, false));

            const ArrayInfo& cinfo = getInfo(rrT);
            float* pSrc0 = pinnedAlloc<float>(cinfo.elements());
            float* pSrc1 = pinnedAlloc<float>(cinfo.elements());
            float* pSrc2 = pinnedAlloc<float>(cinfo.elements());
            float* pSrc3 = pinnedAlloc<float>(cinfo.elements());

            AF_CHECK(af_get_data_ptr((void*)pSrc0, rrT));
            AF_CHECK(af_get_data_ptr((void*)pSrc1, ggT));
            AF_CHECK(af_get_data_ptr((void*)pSrc2, bbT));
            AF_CHECK(af_get_data_ptr((void*)pSrc3, aaT));

            // Copy the array into FreeImage buffer
            for (uint y = 0; y < fi_h; ++y) {
                for (uint x = 0; x < fi_w; ++x) {
                    *(pDstLine + x * step + FI_RGBA_RED  ) = (uchar) pSrc0[indx]; // r
                    *(pDstLine + x * step + FI_RGBA_GREEN) = (uchar) pSrc1[indx]; // g
                    *(pDstLine + x * step + FI_RGBA_BLUE ) = (uchar) pSrc2[indx]; // b
                    *(pDstLine + x * step + FI_RGBA_ALPHA) = (uchar) pSrc3[indx]; // a
                    ++indx;
                }
                pDstLine -= nDstPitch;
            }
            pinnedFree(pSrc0);
            pinnedFree(pSrc1);
            pinnedFree(pSrc2);
            pinnedFree(pSrc3);
        } else if(channels == 3) {
            AF_CHECK(af_transpose(&rrT, rr, false));
            AF_CHECK(af_transpose(&ggT, gg, false));
            AF_CHECK(af_transpose(&bbT, bb, false));

            const ArrayInfo& cinfo = getInfo(rrT);
            float* pSrc0 = pinnedAlloc<float>(cinfo.elements());
            float* pSrc1 = pinnedAlloc<float>(cinfo.elements());
            float* pSrc2 = pinnedAlloc<float>(cinfo.elements());

            AF_CHECK(af_get_data_ptr((void*)pSrc0, rrT));
            AF_CHECK(af_get_data_ptr((void*)pSrc1, ggT));
            AF_CHECK(af_get_data_ptr((void*)pSrc2, bbT));

            // Copy the array into FreeImage buffer
            for (uint y = 0; y < fi_h; ++y) {
                for (uint x = 0; x < fi_w; ++x) {
                    *(pDstLine + x * step + FI_RGBA_RED  ) = (uchar) pSrc0[indx]; // r
                    *(pDstLine + x * step + FI_RGBA_GREEN) = (uchar) pSrc1[indx]; // g
                    *(pDstLine + x * step + FI_RGBA_BLUE ) = (uchar) pSrc2[indx]; // b
                    ++indx;
                }
                pDstLine -= nDstPitch;
            }
            pinnedFree(pSrc0);
            pinnedFree(pSrc1);
            pinnedFree(pSrc2);
        } else {
            AF_CHECK(af_transpose(&rrT, rr, false));
            const ArrayInfo& cinfo = getInfo(rrT);
            float* pSrc0 = pinnedAlloc<float>(cinfo.elements());
            AF_CHECK(af_get_data_ptr((void*)pSrc0, rrT));

            for (uint y = 0; y < fi_h; ++y) {
                for (uint x = 0; x < fi_w; ++x) {
                    *(pDstLine + x * step) = (uchar) pSrc0[indx];
                    ++indx;
                }
                pDstLine -= nDstPitch;
            }
            pinnedFree(pSrc0);
        }

        FIMEMORY *stream = FreeImage_OpenMemory();

        int flags = 0;
        if(fif == FIF_JPEG) flags = flags | JPEG_QUALITYSUPERB;

        // now save the result image
        if (!(FreeImage_SaveToMemory(fif, bmp, stream, flags) == TRUE)) {
            AF_ERROR("FreeImage Error: Failed to save image", AF_ERR_RUNTIME);
        }

        *ptr = stream;

        if(free_in) AF_CHECK(af_release_array(in ));
        if(rr != 0) AF_CHECK(af_release_array(rr ));
        if(gg != 0) AF_CHECK(af_release_array(gg ));
        if(bb != 0) AF_CHECK(af_release_array(bb ));
        if(aa != 0) AF_CHECK(af_release_array(aa ));
        if(rrT!= 0) AF_CHECK(af_release_array(rrT));
        if(ggT!= 0) AF_CHECK(af_release_array(ggT));
        if(bbT!= 0) AF_CHECK(af_release_array(bbT));
        if(aaT!= 0) AF_CHECK(af_release_array(aaT));

    } CATCHALL

    return AF_SUCCESS;
}

af_err af_delete_image_memory(void *ptr)
{
    try {

        ARG_ASSERT(0, ptr != NULL);

        FI_Init();

        FIMEMORY *stream = (FIMEMORY*)ptr;
        FreeImage_SeekMemory(stream, 0L, SEEK_SET);

        // Ensure data is freeimage compatible
        FREE_IMAGE_FORMAT fif = FreeImage_GetFileTypeFromMemory((FIMEMORY*)ptr, 0);
        if(fif == FIF_UNKNOWN) {
            AF_ERROR("FreeImage Error: Unknown Filetype", AF_ERR_NOT_SUPPORTED);
        }

        FreeImage_CloseMemory((FIMEMORY *)ptr);

    } CATCHALL

    return AF_SUCCESS;
}

#else   // WITH_FREEIMAGE
#include <af/image.h>
#include <stdio.h>
#include <err_common.hpp>
af_err af_load_image(af_array *out, const char* filename, const bool isColor)
{
    AF_RETURN_ERROR("ArrayFire compiled without Image IO (FreeImage) support", AF_ERR_NOT_CONFIGURED);
}

af_err af_save_image(const char* filename, const af_array in_)
{
    AF_RETURN_ERROR("ArrayFire compiled without Image IO (FreeImage) support", AF_ERR_NOT_CONFIGURED);
}

af_err af_load_image_memory(af_array *out, const void* ptr)
{
    AF_RETURN_ERROR("ArrayFire compiled without Image IO (FreeImage) support", AF_ERR_NOT_CONFIGURED);
}

af_err af_save_image_memory(void **ptr, const af_array in_, const af_image_format format)
{
    AF_RETURN_ERROR("ArrayFire compiled without Image IO (FreeImage) support", AF_ERR_NOT_CONFIGURED);
}

af_err af_delete_image_memory(void *ptr)
{
    AF_RETURN_ERROR("ArrayFire compiled without Image IO (FreeImage) support", AF_ERR_NOT_CONFIGURED);
}
#endif  // WITH_FREEIMAGE
