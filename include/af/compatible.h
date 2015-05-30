/*******************************************************
 * Copyright (c) 2014, ArrayFire
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

/// \ingroup device_func_count
/// \copydoc getDeviceCount()
/// \deprecated Use getDeviceCount() instead
DEPRECATED("Use getDeviceCount instead")
AFAPI int devicecount();

/// \ingroup device_func_get
/// \copydoc getDevice()
/// \deprecated Use getDevice() instead
DEPRECATED("Use getDevice instead")
AFAPI int deviceget();

/// \ingroup device_func_set
/// \copydoc setDevice()
/// \deprecated Use setDevice() instead
DEPRECATED("Use setDevice instead")
AFAPI void deviceset(const int device);

/// \ingroup imageio_func_load
/// \copydoc loadImage
/// \deprecated Use \ref loadImage instead
DEPRECATED("Use loadImage instead")
AFAPI array loadimage(const char* filename, const bool is_color=false);

/// \ingroup imageio_func_save
/// \copydoc saveImage
/// \deprecated Use \ref saveImage instead
DEPRECATED("Use saveImage instead")
AFAPI void saveimage(const char* filename, const array& in);

/// \ingroup image_func_gauss
/// \copydoc image_func_gauss
/// \deprecated Use \ref gaussianKernel instead
DEPRECATED("Use gaussianKernel instead")
AFAPI array gaussiankernel(const int rows, const int cols, const double sig_r = 0, const double sig_c = 0);

/// \ingroup reduce_func_all_true
/// \copydoc af::allTrue(const array&)
/// \deprecated Use \ref af::allTrue(const array&) instead
template<typename T>
DEPRECATED("Use allTrue instead")
T alltrue(const array &in);

/// \ingroup reduce_func_any_true
/// \copydoc af::allTrue(const array&)
/// \deprecated Use \ref af::anyTrue(const array&) instead
template<typename T>
DEPRECATED("Use anyTrue instead")
T anytrue(const array &in);

/// \ingroup reduce_func_all_true
/// \copydoc allTrue
/// \deprecated Use \ref af::allTrue instead
DEPRECATED("Use allTrue instead")
AFAPI array alltrue(const array &in, const int dim = -1);

/// \ingroup reduce_func_any_true
/// \copydoc anyTrue
/// \deprecated Use \ref af::anyTrue instead
DEPRECATED("Use anyTrue instead")
AFAPI array anytrue(const array &in, const int dim = -1);

/// \ingroup set_func_unique
/// \copydoc setUnique
/// \deprecated Use \ref setUnique instead
DEPRECATED("Use setUnique instead")
AFAPI array setunique(const array &in, const bool is_sorted=false);

/// \ingroup set_func_union
/// \copydoc setUnion
/// \deprecated Use \ref setUnion instead
DEPRECATED("Use setUnion instead")
AFAPI array setunion(const array &first, const array &second, const bool is_unique=false);

/// \ingroup set_func_intersect
/// \copydoc setIntersect
/// \deprecated Use \ref setIntersect instead
DEPRECATED("Use setIntersect instead")
AFAPI array setintersect(const array &first, const array &second, const bool is_unique=false);

/// \ingroup image_func_histequal
/// \copydoc histEqual
/// \deprecated Use \ref histEqual instead
DEPRECATED("Use histEqual instead")
AFAPI array histequal(const array& in, const array& hist);

/// \ingroup image_func_colorspace
/// \copydoc colorSpace
/// \deprecated Use \ref colorSpace instead
DEPRECATED("Use colorSpace instead")
AFAPI array colorspace(const array& image, const CSpace to, const CSpace from);

/// Image Filtering
/// \code
/// // filter (convolve) an image with a 3x3 sobel kernel
/// const float h_kernel[] = { -2.0, -1.0,  0.0,
///                            -1.0,  0.0,  1.0,
///                             0.0,  1.0,  2.0 };
/// array kernel = array(3,3,h_kernel);
/// array img_out = filter(img_in, kernel);
/// \endcode
///
/// \param[in] image
/// \param[in] kernel coefficient matrix
/// \returns filtered image (same size as input)
///
/// \note Filtering done using correlation. Array values outside bounds are assumed to have zero value (0).
/// \ingroup image_func_filter
/// \deprecated Use \ref af::convolve instead
DEPRECATED("Use af::convolve instead")
AFAPI array filter(const array& image, const array& kernel);

/// \ingroup reduce_func_product
/// \copydoc product(const array&, const int);
/// \deprecated Use \ref product instead
DEPRECATED("Use af::product instead")
AFAPI array mul(const array& in, const int dim = -1);

/// \ingroup reduce_func_product
/// \copydoc product(const array&)
/// \deprecated Use \ref product instead
template<typename T>
DEPRECATED("Use af::product instead")
T mul(const array& in);

/// \ingroup device_func_prop
/// \copydoc deviceInfo
/// \deprecated Use \ref deviceInfo instead
DEPRECATED("Use deviceInfo instead")
AFAPI void deviceprop(char* d_name, char* d_platform, char *d_toolkit, char* d_compute);

}
#endif
