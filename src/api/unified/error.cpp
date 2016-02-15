/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/exception.h>
#include "symbol_manager.hpp"

void af_get_last_error(char **str, dim_t *len)
{
    typedef void(*af_func)(char **, dim_t *);
    af_func func = (af_func)LOAD_SYMBOL();
    return func(str, len);
}

const char *af_err_to_string(const af_err err)
{
    typedef char *(*af_func)(af_err);
    af_func func = (af_func)LOAD_SYMBOL();
    return func(err);
}
