/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/compatible.h>
#include <af/image.h>
#include "error.hpp"

namespace af {

array loadImage(const char* filename, const bool is_color) {
    af_array out = 0;
    AF_THROW(af_load_image(&out, filename, is_color));
    return array(out);
}

array loadImageMem(const void* ptr) {
    af_array out = 0;
    AF_THROW(af_load_image_memory(&out, ptr));
    return array(out);
}

array loadimage(const char* filename, const bool is_color) {
    return loadImage(filename, is_color);
}

void saveImage(const char* filename, const array& in) {
    AF_THROW(af_save_image(filename, in.get()));
}

void* saveImageMem(const array& in, const imageFormat format) {
    void* ptr = NULL;
    AF_THROW(af_save_image_memory(&ptr, in.get(), format));
    return ptr;
}

void saveimage(const char* filename, const array& in) {
    return saveImage(filename, in);
}

void deleteImageMem(void* ptr) { AF_THROW(af_delete_image_memory(ptr)); }

array loadImageNative(const char* filename) {
    af_array out = 0;
    AF_THROW(af_load_image_native(&out, filename));
    return array(out);
}

void saveImageNative(const char* filename, const array& in) {
    AF_THROW(af_save_image_native(filename, in.get()));
}

bool isImageIOAvailable() {
    bool out = false;
    AF_THROW(af_is_image_io_available(&out));
    return out;
}

}  // namespace af
