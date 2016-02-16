/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/exception.h>
#include <af/device.h>
#include <err_common.hpp>
#include <string>
#include <algorithm>

void af_get_last_error(char **str, dim_t *len)
{
    std::string &global_error_string = get_global_error_string();
    dim_t slen = std::min(MAX_ERR_SIZE, (int)global_error_string.size());

    if (len && slen == 0) {
        *len = 0;
        *str = NULL;
        return;
    }

    af_alloc_host((void**)str, sizeof(char) * (slen + 1));
    global_error_string.copy(*str, slen);

    (*str)[slen] = '\0';
    global_error_string = std::string("");

    if(len) *len = slen;
}

const char *af_err_to_string(const af_err err)
{
    switch (err) {
    case AF_SUCCESS:                return "Success";
    case AF_ERR_NO_MEM:             return "Device out of memory";
    case AF_ERR_DRIVER:             return "Driver not available or incompatible";
    case AF_ERR_RUNTIME:            return "Runtime error ";
    case AF_ERR_INVALID_ARRAY:      return "Invalid array";
    case AF_ERR_ARG:                return "Invalid input argument";
    case AF_ERR_SIZE:               return "Invalid input size";
    case AF_ERR_TYPE:               return "Function does not support this data type";
    case AF_ERR_DIFF_TYPE:          return "Input types are not the same";
    case AF_ERR_BATCH:              return "Invalid batch configuration";
    case AF_ERR_NOT_SUPPORTED:      return "Function not supported";
    case AF_ERR_NOT_CONFIGURED:     return "Function not configured to build";
    case AF_ERR_NONFREE:            return "Function unavailable. "
                                           "ArrayFire compiled without Non-Free algorithms support";
    case AF_ERR_NO_DBL:             return "Double precision not supported for this device";
    case AF_ERR_NO_GFX:             return "Graphics functionality unavailable. "
                                           "ArrayFire compiled without Graphics support";
    case AF_ERR_LOAD_LIB:           return "Failed to load dynamic library. ";
    case AF_ERR_LOAD_SYM:           return "Failed to load symbol";
    case AF_ERR_ARR_BKND_MISMATCH:  return "There was a mismatch between an array and the current backend";
    case AF_ERR_INTERNAL:           return "Internal error";
    case AF_ERR_UNKNOWN:
    default:                        return "Unknown error";
    }
}
