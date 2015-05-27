/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/features.h>
#include <af/array.h>
#include <features.hpp>
#include <handle.hpp>

af_err af_release_features(af_features featHandle)
{

    try {
        af_features_t feat = *(af_features_t *)featHandle;
        if (feat.n > 0) {
            if (feat.x != 0)           AF_CHECK(af_release_array(feat.x));
            if (feat.y != 0)           AF_CHECK(af_release_array(feat.y));
            if (feat.score != 0)       AF_CHECK(af_release_array(feat.score));
            if (feat.orientation != 0) AF_CHECK(af_release_array(feat.orientation));
            if (feat.size != 0)        AF_CHECK(af_release_array(feat.size));
            feat.n = 0;
        }
        delete (af_features_t *)featHandle;
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_features getFeaturesHandle(const af_features_t feat)
{
    af_features_t *featHandle = new af_features_t;
    *featHandle = feat;
    return (af_features)featHandle;
}

af_err af_create_features(af_features *featHandle, dim_t num)
{
    try {
        af_features_t feat;
        feat.n = num;

        if (num > 0) {
            dim_t out_dims[4] = {dim_t(num), 1, 1, 1};
            AF_CHECK(af_create_handle(&feat.x, 4, out_dims, f32));
            AF_CHECK(af_create_handle(&feat.y, 4, out_dims, f32));
            AF_CHECK(af_create_handle(&feat.score, 4, out_dims, f32));
            AF_CHECK(af_create_handle(&feat.orientation, 4, out_dims, f32));
            AF_CHECK(af_create_handle(&feat.size, 4, out_dims, f32));
        }

        *featHandle = getFeaturesHandle(feat);
    } CATCHALL;

    return AF_SUCCESS;
}

af_features_t getFeatures(const af_features featHandle)
{
    return *(af_features_t *)featHandle;
}

af_err af_retain_features(af_features *outHandle, const af_features featHandle)
{
    try {

        af_features_t feat = getFeatures(featHandle);
        af_features_t out;

        out.n = feat.n;
        AF_CHECK(af_retain_array(&out.x, feat.x));
        AF_CHECK(af_retain_array(&out.y, feat.y));
        AF_CHECK(af_retain_array(&out.score, feat.score));
        AF_CHECK(af_retain_array(&out.orientation, feat.orientation));
        AF_CHECK(af_retain_array(&out.size, feat.size));

        *outHandle = getFeaturesHandle(out);

    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_features_num(dim_t *num, const af_features featHandle)
{
    try {

        af_features_t feat = getFeatures(featHandle);
        *num = feat.n;

    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_features_xpos(af_array *out, const af_features featHandle)
{
    try {

        af_features_t feat = getFeatures(featHandle);
        *out = feat.x;
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_features_ypos(af_array *out, const af_features featHandle)
{
    try {

        af_features_t feat = getFeatures(featHandle);
        *out = feat.y;
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_features_score(af_array *out, const af_features featHandle)
{
    try {

        af_features_t feat = getFeatures(featHandle);
        *out = feat.score;
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_features_orientation(af_array *out, const af_features featHandle)
{
    try {

        af_features_t feat = getFeatures(featHandle);
        *out = feat.orientation;
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_features_size(af_array *out, const af_features featHandle)
{
    try {

        af_features_t feat = getFeatures(featHandle);
        *out = feat.size;
    } CATCHALL;
    return AF_SUCCESS;
}
