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
#include <af/array.h>
#include <af/dim4.hpp>
#include <af/seq.h>
#include <af/traits.hpp>

typedef struct {
    size_t n;
    af_array x;
    af_array y;
    af_array score;
    af_array orientation;
    af_array size;
} af_features;

#ifdef __cplusplus
namespace af
{

    class AFAPI features {
    private:
        af_features feat;

    public:
        features();
        features(const size_t n);
        features(af_features f);

        ~features();

        features& operator= (const features& f);

        size_t getNumFeatures() const;
        array getX() const;
        array getY() const;
        array getScore() const;
        array getOrientation() const;
        array getSize() const;
        af_features get() const;
    };

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    // Destroy af_features
    AFAPI af_err af_destroy_features(af_features feat);

#ifdef __cplusplus
}
#endif
