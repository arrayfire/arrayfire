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

        size_t getNumFeatures();
        array getX();
        array getY();
        array getScore();
        array getOrientation();
        array getSize();

        void setNumFeatures(const size_t n);
        void setX(const array x);
        void setX(const af_array x);
        void setY(const array y);
        void setY(const af_array y);
        void setScore(const array score);
        void setScore(const af_array score);
        void setOrientation(const array orientation);
        void setOrientation(const af_array orientation);
        void setSize(const array size);
        void setSize(const af_array size);

        af_features get();
    };

}
#endif
