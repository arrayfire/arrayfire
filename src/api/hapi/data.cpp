/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/data.h>
#include <functional>
#include <stdlib.h>
#include <dlfcn.h>
#include <iostream>
#include "symbol_manager.hpp"

af_err af_create_array(af_array *result, const void * const data,
                       const unsigned ndims, const dim_t * const dims,
                       const af_dtype type)
{
    af_err errCode = AF_SUCCESS;
    try {
        AFSymbolManager& symbolManager = AFSymbolManager::getInstance();
        errCode = symbolManager.call("af_create_array", result, data, ndims, dims, type);
    } catch(std::logic_error &e) {
        // FIXME: remove std::cerr
        std::cerr<<e.what()<<std::endl;
        errCode = AF_ERR_SYM_LOAD;
    }
    return errCode;
}

af_err af_constant(af_array *result, const double value,
                   const unsigned ndims, const dim_t * const dims,
                   const af_dtype type)
{
    af_err errCode = AF_SUCCESS;
    try {
        AFSymbolManager& symbolManager = AFSymbolManager::getInstance();
        errCode = symbolManager.call("af_constant", result, value, ndims, dims, type);
    } catch(std::logic_error &e) {
        // FIXME: remove std::cerr
        std::cerr<<e.what()<<std::endl;
        errCode = AF_ERR_SYM_LOAD;
    }
    return errCode;
}

af_err af_release_array(af_array arr)
{
    af_err errCode = AF_SUCCESS;
    try {
        AFSymbolManager& symbolManager = AFSymbolManager::getInstance();
        errCode = symbolManager.call("af_release_array", arr);
    } catch(std::logic_error &e) {
        // FIXME: remove std::cerr
        std::cerr<<e.what()<<std::endl;
        errCode = AF_ERR_SYM_LOAD;
    }
    return errCode;
}
