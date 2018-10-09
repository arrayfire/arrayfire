/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/ml.h>
#include "symbol_manager.hpp"

af_err af_pool2(af_array *out, const af_array in, const dim_t pool_width,
                const dim_t pool_height, const dim_t padding_width,
                const dim_t padding_height, const dim_t stride_width,
                const dim_t stride_height, af_pooling_type pool_type) {
    CHECK_ARRAYS(in);
    return CALL(out, in, pool_width, pool_height, stride_width, stride_height,
                pool_type);
}

af_err af_pool2Gradient(af_array *out, const af_array original_input,
                        const af_array pooled_output,
                        const af_array incoming_gradient,
                        const dim_t pool_width, const dim_t pool_height,
                        const dim_t padding_width, const dim_t padding_height,
                        const dim_t stride_width, const dim_t stride_height,
                        af_pooling_type pool_type = af_pooling_max) {
    CHECK_ARRAYS(incoming_gradient);
    CHECK_ARRAYS(original_input);
    CHECK_ARRAYS(pooled_output);
    return CALL(out, incoming_gradient, original_input, pooled_output,
                pool_width, pool_height, padding_width, padding_height,
                stride_width, stride_height, pool_type);
}
