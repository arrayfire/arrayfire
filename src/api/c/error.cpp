/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/err_common.hpp>
#include <af/device.h>
#include <af/exception.h>
#include <af/util.h>

#include <algorithm>
#include <cstring>
#include <string>

void af_get_last_error(char **str, dim_t *len) {
    std::string &global_error_string = get_global_error_string();
    dim_t slen =
        std::min(MAX_ERR_SIZE, static_cast<int>(global_error_string.size()));

    if (len && slen == 0) {
        *len = 0;
        *str = NULL;
        return;
    }

    void *halloc_ptr = nullptr;
    af_alloc_host(&halloc_ptr, sizeof(char) * (slen + 1));
    memcpy(str, &halloc_ptr, sizeof(void *));
    global_error_string.copy(*str, slen);

    (*str)[slen]        = '\0';
    global_error_string = std::string("");

    if (len) { *len = slen; }
}

af_err af_set_enable_stacktrace(int is_enabled) {
    arrayfire::common::is_stacktrace_enabled() = is_enabled;

    return AF_SUCCESS;
}
