/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/util.h>
#include <version.hpp>

af_err af_get_version(int *major, int *minor, int *patch)
{
    *major = AF_VERSION_MAJOR;
    *minor = AF_VERSION_MINOR;
    *patch = AF_VERSION_PATCH;

    return AF_SUCCESS;
}

const char *af_get_revision()
{
    return AF_REVISION;
}
