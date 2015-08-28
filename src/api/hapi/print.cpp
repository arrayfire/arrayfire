/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/util.h>
#include "symbol_manager.hpp"

af_err af_print_array(const af_array arr)
{
    AFSymbolManager& symbolManager = AFSymbolManager::getInstance();
    return symbolManager.call("af_print_array", arr);
}
