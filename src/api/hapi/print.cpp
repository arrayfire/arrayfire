/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/util.h>
#include <functional>
#include <stdlib.h>
#include <dlfcn.h>
#include <iostream>
#include "symbol_manager.hpp"

af_err af_print_array(const af_array arr)
{
    af_err errCode = AF_SUCCESS;
    try {
        AFSymbolManager& symbolManager = AFSymbolManager::getInstance();
        errCode = symbolManager.call("af_print_array", arr);
    } catch(std::logic_error &e) {
        // FIXME: remove std::cerr
        std::cerr<<e.what()<<std::endl;
        errCode = AF_ERR_SYM_LOAD;
    }
    return errCode;
}
