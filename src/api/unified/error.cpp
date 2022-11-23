/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/device.h>
#include <af/exception.h>
#include <af/util.h>
#include <algorithm>
#include "symbol_manager.hpp"

void af_get_last_error(char **str, dim_t *len) {
    // Set error message from unified backend
    std::string &global_error_string = get_global_error_string();
    dim_t slen =
        std::min(MAX_ERR_SIZE, static_cast<int>(global_error_string.size()));

    // If this is true, the error is coming from the unified backend.
    if (slen != 0) {
        if (len && slen == 0) {
            *len = 0;
            *str = NULL;
            return;
        }

        void *in = nullptr;
        af_alloc_host(&in, sizeof(char) * (slen + 1));
        memcpy(str, &in, sizeof(void *));
        global_error_string.copy(*str, slen);

        (*str)[slen]        = '\0';
        global_error_string = std::string("");

        if (len) { *len = slen; }
    } else {
        // If false, the error is coming from active backend.
        typedef void (*af_func)(char **, dim_t *);
        void *vfn    = LOAD_SYMBOL();
        af_func func = nullptr;
        memcpy(&func, vfn, sizeof(void *));
        func(str, len);
    }
}

af_err af_set_enable_stacktrace(int is_enabled) {
    CALL(af_set_enable_stacktrace, is_enabled);
}
