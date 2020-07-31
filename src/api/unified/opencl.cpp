/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/backend.h>
#include "symbol_manager.hpp"

#include <af/opencl.h>

af_err afcl_get_device_type(afcl_device_type* res) {
    af_backend backend;
    af_get_active_backend(&backend);
    if (backend == AF_BACKEND_OPENCL) { CALL(afcl_get_device_type, res); }
    return AF_ERR_NOT_SUPPORTED;
}

af_err afcl_get_platform(afcl_platform* res) {
    af_backend backend;
    af_get_active_backend(&backend);
    if (backend == AF_BACKEND_OPENCL) { CALL(afcl_get_platform, res); }
    return AF_ERR_NOT_SUPPORTED;
}

af_err afcl_get_context(cl_context* ctx, const bool retain) {
    af_backend backend;
    af_get_active_backend(&backend);
    if (backend == AF_BACKEND_OPENCL) { CALL(afcl_get_context, ctx, retain); }
    return AF_ERR_NOT_SUPPORTED;
}

af_err afcl_get_queue(cl_command_queue* queue, const bool retain) {
    af_backend backend;
    af_get_active_backend(&backend);
    if (backend == AF_BACKEND_OPENCL) { CALL(afcl_get_queue, queue, retain); }
    return AF_ERR_NOT_SUPPORTED;
}

af_err afcl_get_device_id(cl_device_id* id) {
    af_backend backend;
    af_get_active_backend(&backend);
    if (backend == AF_BACKEND_OPENCL) { CALL(afcl_get_device_id, id); }
    return AF_ERR_NOT_SUPPORTED;
}

af_err afcl_set_device_id(cl_device_id id) {
    af_backend backend;
    af_get_active_backend(&backend);
    if (backend == AF_BACKEND_OPENCL) { CALL(afcl_set_device_id, id); }
    return AF_ERR_NOT_SUPPORTED;
}

af_err afcl_add_device_context(cl_device_id dev, cl_context ctx,
                               cl_command_queue que) {
    af_backend backend;
    af_get_active_backend(&backend);
    if (backend == AF_BACKEND_OPENCL) {
        CALL(afcl_add_device_context, dev, ctx, que);
    }
    return AF_ERR_NOT_SUPPORTED;
}

af_err afcl_set_device_context(cl_device_id dev, cl_context ctx) {
    af_backend backend;
    af_get_active_backend(&backend);
    if (backend == AF_BACKEND_OPENCL) {
        CALL(afcl_set_device_context, dev, ctx);
    }
    return AF_ERR_NOT_SUPPORTED;
}

af_err afcl_delete_device_context(cl_device_id dev, cl_context ctx) {
    af_backend backend;
    af_get_active_backend(&backend);
    if (backend == AF_BACKEND_OPENCL) {
        CALL(afcl_delete_device_context, dev, ctx);
    }
    return AF_ERR_NOT_SUPPORTED;
}
