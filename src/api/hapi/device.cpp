/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/hapi.h>
#include <af/device.h>
#include "symbol_manager.hpp"

af_err af_set_backend(const af_backend bknd)
{
    return AFSymbolManager::getInstance().setBackend(bknd);
}

af_err af_info()
{
    AFSymbolManager& symbolManager = AFSymbolManager::getInstance();
    return symbolManager.call("af_info");
}
