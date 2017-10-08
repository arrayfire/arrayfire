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
#include <common/err_common.hpp>
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
