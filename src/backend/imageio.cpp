#include <af/array.h>
#include <af/image.h>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <ArrayInfo.hpp>
#include <traits.hpp>

#include <FreeImage.h>
#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#ifndef DEBUG
#define ASSERT(x)
#else
#define ASSERT(x) \
                 if (! (x)) \
                { \
                    std::cout << "ERROR!! Assert " << #x << " failed\n"; \
                    std::cout << " on line " << __LINE__  << "\n";      \
                    std::cout << " in file " << __FILE__ << "\n";       \
                }
#endif

using af::dim4;
using namespace detail;

// Helpers
void FreeImageErrorHandler(FREE_IMAGE_FORMAT oFif, const char* zMessage);

typedef unsigned short ushort;

// Error handler for FreeImage library.
//  In case this handler is invoked, it throws an af exception.
void FreeImageErrorHandler(FREE_IMAGE_FORMAT oFif, const char* zMessage) {
    printf("Error in Image IO: %s", zMessage);
}

//  Split a MxNx3 image into 3 separate channel matrices.
//  Produce 3 channels if needed
static af_err channel_split(const af_array rgb, const af::dim4 dims,
                            af_array *outr, af_array *outg, af_array *outb, af_array *outa)
{
    af_err ret = AF_SUCCESS;
    af_seq idx[4][3] = {{span, span, {0, 0, 1}},
                        {span, span, {1, 1, 1}},
                        {span, span, {2, 2, 1}},
                        {span, span, {3, 3, 1}}
                       };

    if (dims[2] == 4) {
        ret = af_index(outr, rgb, dims.ndims(), idx[0]);
        if(ret != AF_SUCCESS) return ret;
        ret = af_index(outg, rgb, dims.ndims(), idx[1]);
        if(ret != AF_SUCCESS) return ret;
        ret = af_index(outb, rgb, dims.ndims(), idx[2]);
        if(ret != AF_SUCCESS) return ret;
        ret = af_index(outa, rgb, dims.ndims(), idx[3]);
        if(ret != AF_SUCCESS) return ret;
    } else if (dims[2] == 3) {
        ret = af_index(outr, rgb, dims.ndims(), idx[0]);
        if(ret != AF_SUCCESS) return ret;
        ret = af_index(outg, rgb, dims.ndims(), idx[1]);
        if(ret != AF_SUCCESS) return ret;
        ret = af_index(outb, rgb, dims.ndims(), idx[2]);
        if(ret != AF_SUCCESS) return ret;
    } else {
        ret = af_index(outr, rgb, dims.ndims(), idx[0]);
        if(ret != AF_SUCCESS) return ret;
    }
    return ret;
}

template<typename T, int fi_color, int fo_color>
static af_err readImage(af_array *rImage, const uchar* pSrcLine, const int nSrcPitch,
                        const uint fi_w, const uint fi_h)
{
    // create an array to receive the loaded image data.
    float* pDst = (float*)malloc(fi_w * fi_h * sizeof(float) * 4); //4 channels is max
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
    free(pDst);
    return err;
}

template<typename T, int fo_color>
static af_err readImage(af_array *rImage, const uchar* pSrcLine, const int nSrcPitch,
                        const uint fi_w, const uint fi_h)
{
    // create an array to receive the loaded image data.
    float* pDst = (float*)malloc(fi_w * fi_h * sizeof(float)); //only gray channel

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
    free(pDst);
    return err;
}

/// Load a gray-scale image from disk.
AFAPI af_err af_load_image(af_array *out, const char* filename, const bool isColor)
{
    af_err ret = AF_SUCCESS;
    try {
        // for statically linked FI
    #if defined(_WIN32) || defined(_MSC_VER)
        FreeImage_Initialise();
    #endif
        // set your own FreeImage error handler
        FreeImage_SetOutputMessage(FreeImageErrorHandler);

        // try to guess the file format from the file extension
        FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(filename);
        if (fif == FIF_UNKNOWN) {
            fif = FreeImage_GetFIFFromFilename(filename);
        }
        //if(fif == FIF_UNKNOWN) THROW("unknown filetype %s", filename);
        if(fif == FIF_UNKNOWN) return AF_ERR_ARG;

        // check that the plugin has reading capabilities ...
        FIBITMAP* pBitmap = NULL;
        if (FreeImage_FIFSupportsReading(fif)) {
            pBitmap = FreeImage_Load(fif, filename);
        }
        //if(pBitmap == NULL) THROW("error reading image %s", filename);
        if(pBitmap == NULL) return AF_ERR_ARG;

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
        if(fi_bpc != 8 && fi_bpc != 16 && fi_bpc != 32)
            return AF_ERR_RUNTIME;//THROW("Bits Per channel not supported");

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
                    ret = readImage<uchar, 4, 4>(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h);
                else if(fi_bpc == 16)
                    ret = readImage<ushort, 4, 4>(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h);
                else if(fi_bpc == 32)
                    ret = readImage<float, 4, 4>(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h);
                else
                    ret = AF_ERR_RUNTIME;
            } else if (fi_color == 1) {
                if(fi_bpc == 8)
                    ret = readImage<uchar, 1, 3>(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h);
                else if(fi_bpc == 16)
                    ret = readImage<ushort, 1, 3>(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h);
                else if(fi_bpc == 32)
                    ret = readImage<float, 1, 3>(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h);
                else
                    ret = AF_ERR_RUNTIME;
            } else {             //3 channel image
                if(fi_bpc == 8)
                    ret = readImage<uchar, 3, 3>(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h);
                else if(fi_bpc == 16)
                    ret = readImage<ushort, 3, 3>(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h);
                else if(fi_bpc == 32)
                    ret = readImage<float, 3, 3>(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h);
                else
                    ret = AF_ERR_RUNTIME;
            }
        } else {                    //output gray irrespective
            if(fi_color == 1) {     //4 channel image
                if(fi_bpc == 8)
                    ret = readImage<uchar, 1>(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h);
                else if(fi_bpc == 16)
                    ret = readImage<ushort, 1>(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h);
                else if(fi_bpc == 32)
                    ret = readImage<float, 1>(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h);
                else
                    ret = AF_ERR_RUNTIME;
            } else if (fi_color == 3 || fi_color == 4) {
                if(fi_bpc == 8)
                    ret = readImage<uchar, 3>(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h);
                else if(fi_bpc == 16)
                    ret = readImage<ushort, 3>(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h);
                else if(fi_bpc == 32)
                    ret = readImage<float, 3>(&rImage, pSrcLine, nSrcPitch, fi_w, fi_h);
                else
                    ret = AF_ERR_RUNTIME;
            }
        }

        if(ret == AF_SUCCESS) {
            std::swap(*out,rImage);
        }
    // for statically linked FI
    #if defined(_WIN32) || defined(_MSC_VER)
        FreeImage_DeInitialise();
    #endif
    }
    CATCHALL;

    return ret;
}

// Save an image to disk.
af_err af_save_image(const char* filename, const af_array in_)
{
    af_err ret = AF_SUCCESS;

    // for statically linked FI
#if defined(_WIN32) || defined(_MSC_VER)
    FreeImage_Initialise();
#endif

    // set your own FreeImage error handler
    FreeImage_SetOutputMessage(FreeImageErrorHandler);

    // try to guess the file format from the file extension
    FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(filename);
    if (fif == FIF_UNKNOWN) {
        fif = FreeImage_GetFIFFromFilename(filename);
    }
    //if(fif == FIF_UNKNOWN) THROW("unknown filetype %s", filename);
    if(fif == FIF_UNKNOWN) return AF_ERR_ARG;

    ArrayInfo info = getInfo(in_);

    // check image color type
    uint channels = info.dims()[2];
    //if(channels  > 4)  THROW("too many channels. Max is 4");
    //if(channels  == 2) THROW("2 channels not supported");
    if(channels  > 4)  return AF_ERR_ARG;
    if(channels  == 2) return AF_ERR_ARG;

    int fi_bpp = channels * 8;

    // sizes
    uint fi_w = info.dims()[1];
    uint fi_h = info.dims()[0];

    // create the result image storage using FreeImage
    FIBITMAP* pResultBitmap = FreeImage_Allocate(fi_w, fi_h, fi_bpp);
    //if(pResultBitmap == NULL) THROW("error creating image %s", filename);
    if(pResultBitmap == NULL)
        return AF_ERR_RUNTIME;

    // FI assumes [0-255]
    // TODO FIXME when max is available
    af_array in = in_;
    //if (af::max<float>(in_) <= 1) { in = in_ * 255.f; }
    //else  { in = in_; }

    // FI = row major | AF = column major
    uint nDstPitch = FreeImage_GetPitch(pResultBitmap);
    uchar* pDstLine = FreeImage_GetBits(pResultBitmap) + nDstPitch * (fi_h - 1);
    af_array rr = 0, gg = 0, bb = 0, aa = 0;
    ret = channel_split(in, info.dims(), &rr, &gg, &bb, &aa); // convert array to 3 channels if needed
    if(ret != AF_SUCCESS)
        return ret;

    uint step = channels; // force 3 channels saving
    uint indx = 0;

    af_array rrT = 0, ggT = 0, bbT = 0, aaT = 0;
    if(channels == 4) {
        ret = af_transpose(&rrT, rr);
        if(ret != AF_SUCCESS) return ret;
        ret = af_transpose(&ggT, gg);
        if(ret != AF_SUCCESS) return ret;
        ret = af_transpose(&bbT, bb);
        if(ret != AF_SUCCESS) return ret;
        ret = af_transpose(&aaT, aa);
        if(ret != AF_SUCCESS) return ret;
        ArrayInfo cinfo = getInfo(rrT);
        float* pSrc0 = new float[cinfo.elements()];
        float* pSrc1 = new float[cinfo.elements()];
        float* pSrc2 = new float[cinfo.elements()];
        float* pSrc3 = new float[cinfo.elements()];
        ret = af_get_data_ptr((void*)pSrc0, rrT);
        if(ret != AF_SUCCESS) return ret;
        ret = af_get_data_ptr((void*)pSrc1, ggT);
        if(ret != AF_SUCCESS) return ret;
        ret = af_get_data_ptr((void*)pSrc2, bbT);
        if(ret != AF_SUCCESS) return ret;
        ret = af_get_data_ptr((void*)pSrc3, aaT);
        if(ret != AF_SUCCESS) return ret;

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
        delete [] pSrc0;
        delete [] pSrc1;
        delete [] pSrc2;
        delete [] pSrc3;
    } else if(channels == 3) {
        af_transpose(&rrT, rr);
        if(ret != AF_SUCCESS) return ret;
        af_transpose(&ggT, gg);
        if(ret != AF_SUCCESS) return ret;
        af_transpose(&bbT, bb);
        if(ret != AF_SUCCESS) return ret;
        ArrayInfo cinfo = getInfo(rrT);
        float* pSrc0 = new float[cinfo.elements()];
        float* pSrc1 = new float[cinfo.elements()];
        float* pSrc2 = new float[cinfo.elements()];
        ret = af_get_data_ptr((void*)pSrc0, rrT);
        if(ret != AF_SUCCESS) return ret;
        ret = af_get_data_ptr((void*)pSrc1, ggT);
        if(ret != AF_SUCCESS) return ret;
        ret = af_get_data_ptr((void*)pSrc2, bbT);
        if(ret != AF_SUCCESS) return ret;

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
        delete [] pSrc0;
        delete [] pSrc1;
        delete [] pSrc2;
    } else {
        af_transpose(&rrT, rr);
        if(ret != AF_SUCCESS) return ret;
        ArrayInfo cinfo = getInfo(rrT);
        float* pSrc0 = new float[cinfo.elements()];
        ret = af_get_data_ptr((void*)pSrc0, rrT);
        if(ret != AF_SUCCESS) return ret;

        for (uint y = 0; y < fi_h; ++y) {
            for (uint x = 0; x < fi_w; ++x) {
                *(pDstLine + x * step) = (uchar) pSrc0[indx];
                ++indx;
            }
            pDstLine -= nDstPitch;
        }
        delete [] pSrc0;
    }
    if(rr != 0) ret = af_destroy_array(rr );
    if(gg != 0) ret = af_destroy_array(gg );
    if(bb != 0) ret = af_destroy_array(bb );
    if(aa != 0) ret = af_destroy_array(aa );
    if(rrT!= 0) ret = af_destroy_array(rrT);
    if(ggT!= 0) ret = af_destroy_array(ggT);
    if(bbT!= 0) ret = af_destroy_array(bbT);
    if(aaT!= 0) ret = af_destroy_array(aaT);
    // now save the result image
    if (!(FreeImage_Save(fif, pResultBitmap, filename, 0) == TRUE)) {
        printf("ERROR: Failed to save result image.\n");
    }

    // for statically linked FI
#if defined(_WIN32) || defined(_MSC_VER)
    FreeImage_DeInitialise();
#endif

    return ret;
}
