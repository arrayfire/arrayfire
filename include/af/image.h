#pragma once
#include <af/array.h>

#ifdef __cplusplus
#include <utility>
namespace af
{

AFAPI array loadImage(const char* filename, const bool is_color=false);

AFAPI void saveImage(const char* filename, const array& in);

AFAPI array resize(const array in, const dim_type odim0, const dim_type odim1, const af_interp_type method=AF_INTERP_NEAREST);

AFAPI array transform(const array& in, const array& transform, const dim_type odim0, const dim_type odim1, const bool inverse=true);

AFAPI array rotate(const array& in, const float theta, const bool crop=true, const bool recenter=true);

AFAPI array translate(const array& in, const float trans0, const float trans1, const dim_type odim0, const dim_type odim1);

AFAPI array scale(const array& in, const float scale0, const float scale1, const dim_type odim0, const dim_type odim1);

AFAPI array skew(const array& in, const float skew0, const float skew1, const dim_type odim0, const dim_type odim1, const bool inverse=true);

AFAPI array bilateral(const array &in, const float spatial_sigma, const float chromatic_sigma, bool is_color=false);

AFAPI array histogram(const array &in, const unsigned nbins, const double minval, const double maxval);

AFAPI array meanshift(const array& in, const float spatial_sigma, const float chromatic_sigma, const unsigned iter, const bool is_color);

AFAPI array medfilt(const array& in, dim_type wind_length, dim_type wind_width, af_pad_type edge_pad);

AFAPI array dilate(const array& in, const array& mask);

AFAPI array dilate3d(const array& in, const array& mask);

AFAPI array erode(const array& in, const array& mask);

AFAPI array erode3d(const array& in, const array& mask);

AFAPI std::pair<array, array> gradient(const array& in);

}
#endif

#ifdef __cplusplus
extern "C" {
#endif
    // Image IO: Load and Save Image functions
    AFAPI af_err af_load_image(af_array *out, const char* filename, const bool isColor);

    AFAPI af_err af_save_image(const char* filename, const af_array in);

    // Resize an image/matrix/array
    AFAPI af_err af_resize(af_array *out, const af_array in, const dim_type odim0, const dim_type odim1, const af_interp_type method);

    // Transform an image using a 3x2 transformation matrix.
    // If the transform matrix is a forward transformation matrix, then inverse is false.
    // If the transform martix is an inverse transformation matrix, then inverse is true;
    AFAPI af_err af_transform(af_array *out, const af_array in, const af_array transform, const dim_type odim0, const dim_type odim1, const bool inverse);

    // Rotate
    AFAPI af_err af_rotate(af_array *out, const af_array in, const float theta, const bool crop, const bool recenter);
    // Translate
    AFAPI af_err af_translate(af_array *out, const af_array in, const float trans0, const float trans1,
                              const dim_type odim0, const dim_type odim1);
    // Scale
    AFAPI af_err af_scale(af_array *out, const af_array in, const float scale0, const float scale1,
                          const dim_type odim0, const dim_type odim1);
    // Skew
    AFAPI af_err af_skew(af_array *out, const af_array in, const float skew0, const float skew1,
                         const dim_type odim0, const dim_type odim1, const bool inverse);

    // histogram: return af_array will have elements of type u32
    AFAPI af_err af_histogram(af_array *out, const af_array in, const unsigned nbins, const double minval, const double maxval);

    // image dilation operation
    AFAPI af_err af_dilate(af_array *out, const af_array in, const af_array mask);

    AFAPI af_err af_dilate3d(af_array *out, const af_array in, const af_array mask);

    // image erosion operation
    AFAPI af_err af_erode(af_array *out, const af_array in, const af_array mask);

    AFAPI af_err af_erode3d(af_array *out, const af_array in, const af_array mask);

    // image bilateral filter
    AFAPI af_err af_bilateral(af_array *out, const af_array in, const float spatial_sigma, const float chromatic_sigma, const bool isColor);

    // image meanshift filter
    AFAPI af_err af_meanshift(af_array *out, const af_array in, const float spatial_sigma, const float chromatic_sigma, const unsigned iter, const bool is_color);

    // gradient
    AFAPI af_err af_gradient(af_array *grad_rows, af_array *grad_cols, const af_array in);

    // image median filter

    AFAPI af_err af_medfilt(af_array *out, const af_array in, dim_type wind_length, dim_type wind_width, af_pad_type edge_pad);

#ifdef __cplusplus
}
#endif
