/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <af/array.h>
#include <af/ml.h>
#include "symbol_manager.hpp"

af_err af_convolve2_gradient_nn(
    af_array *out, const af_array incoming_gradient,
    const af_array original_signal, const af_array original_filter,
    const af_array convolved_output, const unsigned stride_dims,
    const dim_t *strides, const unsigned padding_dims, const dim_t *paddings,
    const unsigned dilation_dims, const dim_t *dilations,
    af_conv_gradient_type gradType) {
    CHECK_ARRAYS(incoming_gradient, original_signal, original_filter,
                 convolved_output);
    CALL(af_convolve2_gradient_nn, out, incoming_gradient, original_signal,
         original_filter, convolved_output, stride_dims, strides, padding_dims,
         paddings, dilation_dims, dilations, gradType);
}
