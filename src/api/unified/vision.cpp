/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/vision.h>
#include "symbol_manager.hpp"

af_err af_fast(af_features *out, const af_array in, const float thr,
               const unsigned arc_length, const bool non_max,
               const float feature_ratio, const unsigned edge) {
    CHECK_ARRAYS(in);
    CALL(af_fast, out, in, thr, arc_length, non_max, feature_ratio, edge);
}

af_err af_harris(af_features *out, const af_array in,
                 const unsigned max_corners, const float min_response,
                 const float sigma, const unsigned block_size,
                 const float k_thr) {
    CHECK_ARRAYS(in);
    CALL(af_harris, out, in, max_corners, min_response, sigma, block_size,
         k_thr);
}

af_err af_orb(af_features *feat, af_array *desc, const af_array in,
              const float fast_thr, const unsigned max_feat,
              const float scl_fctr, const unsigned levels,
              const bool blur_img) {
    CHECK_ARRAYS(in);
    CALL(af_orb, feat, desc, in, fast_thr, max_feat, scl_fctr, levels,
         blur_img);
}

af_err af_sift(af_features *feat, af_array *desc, const af_array in,
               const unsigned n_layers, const float contrast_thr,
               const float edge_thr, const float init_sigma,
               const bool double_input, const float intensity_scale,
               const float feature_ratio) {
    CHECK_ARRAYS(in);
    CALL(af_sift, feat, desc, in, n_layers, contrast_thr, edge_thr, init_sigma,
         double_input, intensity_scale, feature_ratio);
}

af_err af_gloh(af_features *feat, af_array *desc, const af_array in,
               const unsigned n_layers, const float contrast_thr,
               const float edge_thr, const float init_sigma,
               const bool double_input, const float intensity_scale,
               const float feature_ratio) {
    CHECK_ARRAYS(in);
    CALL(af_gloh, feat, desc, in, n_layers, contrast_thr, edge_thr, init_sigma,
         double_input, intensity_scale, feature_ratio);
}

af_err af_hamming_matcher(af_array *idx, af_array *dist, const af_array query,
                          const af_array train, const dim_t dist_dim,
                          const unsigned n_dist) {
    CHECK_ARRAYS(query, train);
    CALL(af_hamming_matcher, idx, dist, query, train, dist_dim, n_dist);
}

af_err af_nearest_neighbour(af_array *idx, af_array *dist, const af_array query,
                            const af_array train, const dim_t dist_dim,
                            const unsigned n_dist,
                            const af_match_type dist_type) {
    CHECK_ARRAYS(query, train);
    CALL(af_nearest_neighbour, idx, dist, query, train, dist_dim, n_dist,
         dist_type);
}

af_err af_match_template(af_array *out, const af_array search_img,
                         const af_array template_img,
                         const af_match_type m_type) {
    CHECK_ARRAYS(search_img, template_img);
    CALL(af_match_template, out, search_img, template_img, m_type);
}

af_err af_susan(af_features *out, const af_array in, const unsigned radius,
                const float diff_thr, const float geom_thr,
                const float feature_ratio, const unsigned edge) {
    CHECK_ARRAYS(in);
    CALL(af_susan, out, in, radius, diff_thr, geom_thr, feature_ratio, edge);
}

af_err af_dog(af_array *out, const af_array in, const int radius1,
              const int radius2) {
    CHECK_ARRAYS(in);
    CALL(af_dog, out, in, radius1, radius2);
}

af_err af_homography(af_array *H, int *inliers, const af_array x_src,
                     const af_array y_src, const af_array x_dst,
                     const af_array y_dst, const af_homography_type htype,
                     const float inlier_thr, const unsigned iterations,
                     const af_dtype type) {
    CHECK_ARRAYS(x_src, y_src, x_dst, y_dst);
    CALL(af_homography, H, inliers, x_src, y_src, x_dst, y_dst, htype,
         inlier_thr, iterations, type);
}
