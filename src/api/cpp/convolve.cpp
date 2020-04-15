/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/compatible.h>
#include <af/dim4.hpp>
#include <af/ml.h>
#include <af/signal.h>
#include <algorithm>
#include "error.hpp"

namespace af {

array convolve(const array &signal, const array &filter, const convMode mode,
               convDomain domain) {
    unsigned sN = signal.numdims();
    unsigned fN = filter.numdims();

    switch (std::min(sN, fN)) {
        case 1: return convolve1(signal, filter, mode, domain);
        case 2: return convolve2(signal, filter, mode, domain);
        default:
        case 3: return convolve3(signal, filter, mode, domain);
    }
}

array convolve(const array &col_filter, const array &row_filter,
               const array &signal, const convMode mode) {
    af_array out = 0;
    AF_THROW(af_convolve2_sep(&out, col_filter.get(), row_filter.get(),
                              signal.get(), mode));
    return array(out);
}

array convolve1(const array &signal, const array &filter, const convMode mode,
                convDomain domain) {
    af_array out = 0;
    AF_THROW(af_convolve1(&out, signal.get(), filter.get(), mode, domain));
    return array(out);
}

array convolve2(const array &signal, const array &filter, const convMode mode,
                convDomain domain) {
    af_array out = 0;
    AF_THROW(af_convolve2(&out, signal.get(), filter.get(), mode, domain));
    return array(out);
}

array convolve2NN(
    const array &signal, const array &filter,
    const dim4 stride,      // NOLINT(performance-unnecessary-value-param)
    const dim4 padding,     // NOLINT(performance-unnecessary-value-param)
    const dim4 dilation) {  // NOLINT(performance-unnecessary-value-param)
    af_array out = 0;
    AF_THROW(af_convolve2_nn(&out, signal.get(), filter.get(), 2, stride.get(),
                             2, padding.get(), 2, dilation.get()));
    return array(out);
}

array convolve2GradientNN(
    const array &incoming_gradient, const array &original_signal,
    const array &original_filter, const array &convolved_output,
    const dim4 stride,    // NOLINT(performance-unnecessary-value-param)
    const dim4 padding,   // NOLINT(performance-unnecessary-value-param)
    const dim4 dilation,  // NOLINT(performance-unnecessary-value-param)
    af_conv_gradient_type gradType) {
    af_array out = 0;
    AF_THROW(af_convolve2_gradient_nn(
        &out, incoming_gradient.get(), original_signal.get(),
        original_filter.get(), convolved_output.get(), 2, stride.get(), 2,
        padding.get(), 2, dilation.get(), gradType));
    return array(out);
}

array convolve3(const array &signal, const array &filter, const convMode mode,
                convDomain domain) {
    af_array out = 0;
    AF_THROW(af_convolve3(&out, signal.get(), filter.get(), mode, domain));
    return array(out);
}

array filter(const array &image, const array &kernel) {
    return convolve(image, kernel, AF_CONV_DEFAULT, AF_CONV_AUTO);
}

}  // namespace af
