/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(WITH_FREEIMAGE)

#include <af/array.h>
#include <af/arith.h>
#include <af/algorithm.h>
#include <af/blas.h>
#include <af/data.h>
#include <af/image.h>
#include <af/index.h>
#include <err_common.hpp>
#include <backend.hpp>
#include <ArrayInfo.hpp>
#include <traits.hpp>
#include <memory.hpp>

#include <FreeImage.h>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>

using af::dim4;
using namespace detail;

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
    }

    ~FI_Manager()
    {
#ifdef FREEIMAGE_LIB
        FreeImage_DeInitialise();
#endif
    }
};

bool FI_Manager::initialized = false;

static void FI_Init()
{
    static FI_Manager manager = FI_Manager();
}

// Helpers
void FreeImageErrorHandler(FREE_IMAGE_FORMAT oFif, const char* zMessage);

typedef unsigned short ushort;

// Error handler for FreeImage library.
// In case this handler is invoked, it throws an af exception.
void FreeImageErrorHandler(FREE_IMAGE_FORMAT oFif, const char* zMessage)
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

template<typename T, int fi_color, int fo_color>
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

    int offR = 2; int offG = 1; int offB = 0; int offA = 3;
    if (fo_color == 3 && fi_color == 1) {       //Convert gray to color
        offG = 0; offR = 0;
    }
    uint indx = 0;
    uint step = fi_color;

    for (uint x = 0; x < fi_w; ++x) {
        for (uint y = 0; y < fi_h; ++y) {
            const T *src = (T*)(pSrcLine - y * nSrcPitch);
                               pDst2[indx] = (float) *(src + (x * step + offB));
            if (fo_color >= 3) pDst1[indx] = (float) *(src + (x * step + offG));
            if (fo_color >= 3) pDst0[indx] = (float) *(src + (x * step + offR));
            if (fo_color == 4) pDst3[indx] = (float) *(src + (x * step + offA));
            indx++;
        }
    }

    // TODO
    af::dim4 dims(fi_h, fi_w, fo_color, 1);
    af_err err = af_create_array(rImage, pDst, dims.ndims(), dims.get(), (af_dtype) af::dtype_traits<float>::af_type);
    pinnedFree(pDst);
    return err;
}

template<typename T, int fo_color>
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
            if (fo_color == 1) {
                pDst[indx] = (float) *(src + (x * step));
            } else if (fo_color >=3) {
                r = (float) *(src + (x * step + 2));
                g = (float) *(src + (x * step + 1));
                b = (float) *(src + (x * step + 0));
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

/// Load a gray-scale image from disk.
AFAPI af_err af_load_image(af_array *out, const char* filename, const bool isColor)
{
    try {
        ARG_ASSERT(1, filename != NULL);

        // for statically linked FI
        FI_Init();

        // set your own FreeImage error handler
        FreeImage_SetOutputMessage(FreeImageErrorHandler);

        // try to guess the file format from the file extension
        FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(filename);
        if (fif == FIF_UNKNOWN) {
            fif = FreeImage_GetFIFFromFilename(filename);
        }

        if(fif == FIF_UNKNOWN) {
            AF_ERROR("FreeImage Error: Unknown File or Filetype", AF_ERR_NOT_SUPPORTED);
        }

        // check that the plugin has reading capabilities ...
        FIBITMAP* pBitmap = NULL;
        if (FreeImage_FIFSupportsReading(fif)) {
            pBitmap = FreeImage_Load(fif, filename);
        }

        if(pBitmap == NULL) {
            AF_ERROR("FreeImage Error: Error reading image or file does not exist", AF_ERR_RUNTIME);
        }

        // check image color type
        uint color_type = FreeImage_GetColorType(pBitmap);
        const uint fi_bpp = FreeImage_GetBPP(pBitmap);
        //int fi_color = (int)((fi_bpp / 8.0) + 0.5);        //ceil
        int fi_color;
        if      (color_type == 1) fi_color = 1;
        else if (color_type == 2) fi_color = 3;
        else if (color_type == 4) fi_color = 4;
        else                      fi_color = 3;
        const int fi_bpc = fi_bpp / fi_color;
        if(fi_bpc != 8 && fi_bpc != 16 && fi_bpc != 32) {
            AF_ERROR("FreeImage Error: Bits per channel not supported", AF_ERR_NOT_SUPPORTED);
        }

        // sizes
        uint fi_w = FreeImage_GetWidth(pBitmap);
        uint fi_h = FreeImage_GetHeight(pBitmap);

        // FI = row major | AF = column major
        uint nSrcPitch = FreeImage_GetPitch(pBitmap);
        const uchar* pSrcLine = FreeImage_GetBits(pBitmap) + nSrcPitch * (fi_h - 1);

        // result image
        af_array rImage;
        if (isColor) {
            if(fi_color == 4) {     //4 channel image
                if(fi_bpc == 8)
                    AF_CHECK((readImage<uchar, 4, 4>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
                else if(fi_bpc == 16)
                    AF_CHECK((readImage<ushort, 4, 4>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
                else if(fi_bpc == 32)
                    AF_CHECK((readImage<float, 4, 4>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
            } else if (fi_color == 1) {
                if(fi_bpc == 8)
                    AF_CHECK((readImage<uchar, 1, 3>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
                else if(fi_bpc == 16)
                    AF_CHECK((readImage<ushort, 1, 3>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
                else if(fi_bpc == 32)
                    AF_CHECK((readImage<float, 1, 3>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
            } else {             //3 channel image
                if(fi_bpc == 8)
                    AF_CHECK((readImage<uchar, 3, 3>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
                else if(fi_bpc == 16)
                    AF_CHECK((readImage<ushort, 3, 3>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
                else if(fi_bpc == 32)
                    AF_CHECK((readImage<float, 3, 3>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
            }
        } else {                    //output gray irrespective
            if(fi_color == 1) {     //4 channel image
                if(fi_bpc == 8)
                    AF_CHECK((readImage<uchar, 1>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
                else if(fi_bpc == 16)
                    AF_CHECK((readImage<ushort, 1>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
                else if(fi_bpc == 32)
                    AF_CHECK((readImage<float, 1>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
            } else if (fi_color == 3 || fi_color == 4) {
                if(fi_bpc == 8)
                    AF_CHECK((readImage<uchar, 3>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
                else if(fi_bpc == 16)
                    AF_CHECK((readImage<ushort, 3>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
                else if(fi_bpc == 32)
                    AF_CHECK((readImage<float, 3>)(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h));
            }
        }

        FreeImage_Unload(pBitmap);
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

        // set your own FreeImage error handler
        FreeImage_SetOutputMessage(FreeImageErrorHandler);

        // try to guess the file format from the file extension
        FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(filename);
        if (fif == FIF_UNKNOWN) {
            fif = FreeImage_GetFIFFromFilename(filename);
        }

        if(fif == FIF_UNKNOWN) {
            AF_ERROR("FreeImage Error: Unknown Filetype", AF_ERR_NOT_SUPPORTED);
        }

        ArrayInfo info = getInfo(in_);
        // check image color type
        uint channels = info.dims()[2];
        DIM_ASSERT(1, channels <= 4);
        DIM_ASSERT(1, channels != 2);

        int fi_bpp = channels * 8;

        // sizes
        uint fi_w = info.dims()[1];
        uint fi_h = info.dims()[0];

        // create the result image storage using FreeImage
        FIBITMAP* pResultBitmap = FreeImage_Allocate(fi_w, fi_h, fi_bpp);
        if(pResultBitmap == NULL) {
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
        uint nDstPitch = FreeImage_GetPitch(pResultBitmap);
        uchar* pDstLine = FreeImage_GetBits(pResultBitmap) + nDstPitch * (fi_h - 1);
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

            ArrayInfo cinfo = getInfo(rrT);
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
                    *(pDstLine + x * step + 2) = (uchar) pSrc0[indx]; // b
                    *(pDstLine + x * step + 1) = (uchar) pSrc1[indx]; // g
                    *(pDstLine + x * step + 0) = (uchar) pSrc2[indx]; // r
                    *(pDstLine + x * step + 3) = (uchar) pSrc3[indx]; // a
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

            ArrayInfo cinfo = getInfo(rrT);
            float* pSrc0 = pinnedAlloc<float>(cinfo.elements());
            float* pSrc1 = pinnedAlloc<float>(cinfo.elements());
            float* pSrc2 = pinnedAlloc<float>(cinfo.elements());

            AF_CHECK(af_get_data_ptr((void*)pSrc0, rrT));
            AF_CHECK(af_get_data_ptr((void*)pSrc1, ggT));
            AF_CHECK(af_get_data_ptr((void*)pSrc2, bbT));

            // Copy the array into FreeImage buffer
            for (uint y = 0; y < fi_h; ++y) {
                for (uint x = 0; x < fi_w; ++x) {
                    *(pDstLine + x * step + 2) = (uchar) pSrc0[indx]; // b
                    *(pDstLine + x * step + 1) = (uchar) pSrc1[indx]; // g
                    *(pDstLine + x * step + 0) = (uchar) pSrc2[indx]; // r
                    ++indx;
                }
                pDstLine -= nDstPitch;
            }
            pinnedFree(pSrc0);
            pinnedFree(pSrc1);
            pinnedFree(pSrc2);
        } else {
            AF_CHECK(af_transpose(&rrT, rr, false));
            ArrayInfo cinfo = getInfo(rrT);
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

        // now save the result image
        if (!(FreeImage_Save(fif, pResultBitmap, filename, 0) == TRUE)) {
            AF_ERROR("FreeImage Error: Failed to save image", AF_ERR_RUNTIME);
        }

        FreeImage_Unload(pResultBitmap);

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

#else   // WITH_FREEIMAGE
#include <af/image.h>
#include <stdio.h>
AFAPI af_err af_load_image(af_array *out, const char* filename, const bool isColor)
{
    printf("Error: Image IO requires FreeImage. See https://github.com/arrayfire/arrayfire\n");
    return AF_ERR_NOT_CONFIGURED;
}

af_err af_save_image(const char* filename, const af_array in_)
{
    printf("Error: Image IO requires FreeImage. See https://github.com/arrayfire/arrayfire\n");
    return AF_ERR_NOT_CONFIGURED;
}
#endif  // WITH_FREEIMAGE
