/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/features.h>
#include "symbol_manager.hpp"

af_err af_create_features(af_features *feat, dim_t num) {
    CALL(af_create_features, feat, num);
}

af_err af_retain_features(af_features *out, const af_features feat) {
    CALL(af_retain_features, out, feat);
}

af_err af_get_features_num(dim_t *num, const af_features feat) {
    CALL(af_get_features_num, num, feat);
}

#define FEAT_HAPI_DEF(af_func)                              \
    af_err af_func(af_array *out, const af_features feat) { \
        CALL(af_func, out, feat);                           \
    }

FEAT_HAPI_DEF(af_get_features_xpos)
FEAT_HAPI_DEF(af_get_features_ypos)
FEAT_HAPI_DEF(af_get_features_score)
FEAT_HAPI_DEF(af_get_features_orientation)
FEAT_HAPI_DEF(af_get_features_size)

af_err af_release_features(af_features feat) {
    CALL(af_release_features, feat);
}
