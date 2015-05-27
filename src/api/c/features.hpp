#pragma once

typedef struct {
    size_t n;
    af_array x;
    af_array y;
    af_array score;
    af_array orientation;
    af_array size;
} af_features_t;

af_features getFeaturesHandle(const af_features_t feat);

af_features_t getFeatures(const af_features featHandle);
