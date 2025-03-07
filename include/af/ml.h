/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>

#ifdef __cplusplus
namespace af
{
class array;
class dim4;

#if AF_API_VERSION >= 37
    /**
        C++ interface for calculating backward pass gradient of 2D convolution
        This function calculates the gradient with respect to the output
        of the \ref convolve2NN function that uses the machine learning
        formulation for the dimensions of the signals and filters

        \param[in]  incoming_gradient gradients to be distributed in backwards pass
        \param[in]  original_signal input signal to forward pass of convolution
                    assumed structure of input is ( d0 x d1 x d2 x N )
        \param[in]  original_filter input filter to forward pass of convolution
                    assumed structure of input is ( d0 x d1 x d2 x N )
        \param[in]  convolved_output output from forward pass of convolution
        \param[in]  stride specifies strides along each dimension for original convolution
        \param[in]  padding specifies padding width along each dimension for original convolution
        \param[in]  dilation specifies filter dilation along each dimension for original convolution
        \param[in]  grad_type specifies which gradient to return
        \return     gradient wrt/grad_type

        \note Make sure you pass in both dim0, and dim1 in your dim4 arguments. The third
        and fourth dimensions are currently ignored.

        \ingroup ml_convolution
    */
    AFAPI array convolve2GradientNN(const array& incoming_gradient,
                                    const array& original_signal,
                                    const array& original_filter,
                                    const array& convolved_output,
                                    const dim4 stride, const dim4 padding, const dim4 dilation,
                                    convGradientType grad_type);

#endif

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if AF_API_VERSION >= 37
    /**
        C interface for calculating backward pass gradient of 2D convolution
        This function calculates the gradient with respect to the output
        of the \ref af::convolve2NN() function that uses the machine learning
        formulation for the dimensions of the signals and filters

        \param[out] out gradient wrt/gradType
        \param[in]  incoming_gradient gradients to be distributed in backwards pass
        \param[in]  original_signal input signal to forward pass of convolution
                    assumed structure of input is ( d0 x d1 x d2 x N )
        \param[in]  original_filter input filter to forward pass of convolution
                    assumed structure of input is ( d0 x d1 x d2 x N )
        \param[in]  convolved_output output from forward pass of convolution
        \param[in]  stride_dims specifies number of stride dimensions
        \param[in]  strides array of stride values
        \param[in]  padding_dims number of padding dimensions
        \param[in]  paddings array of padding values
        \param[in]  dilation_dims number of dilation dimensions
        \param[in]  dilations array of dilation values
        \param[in]  grad_type specifies which gradient to return
        \return     \ref AF_SUCCESS if the execution completes properly

        \ingroup ml_convolution
    */
    AFAPI af_err af_convolve2_gradient_nn(af_array *out,
                                          const af_array incoming_gradient,
                                          const af_array original_signal,
                                          const af_array original_filter,
                                          const af_array convolved_output,
                                          const unsigned stride_dims,   const dim_t *strides,
                                          const unsigned padding_dims,  const dim_t *paddings,
                                          const unsigned dilation_dims, const dim_t *dilations,
                                          af_conv_gradient_type grad_type);
#endif


#ifdef __cplusplus
}
#endif
