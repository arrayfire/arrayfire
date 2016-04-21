/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

struct IndexPair
{
    Tk first;
    Tv second;
};

typedef struct IndexPair IndexPair_t;

__kernel
void make_pair_kernel(__global IndexPair_t *out,
                      __global const Tk *first, __global const Tv *second,
                      const unsigned N)
{
    int tIdx = get_group_id(0) * get_local_size(0) * copyPairIter + get_local_id(0);
    const int blockDimX = get_local_size(0);

    for(int i = tIdx; i < N; i += blockDimX) {
        out[i].first  = first[i];
        out[i].second = second[i];
    }
}

__kernel
void split_pair_kernel( __global Tk *first, __global Tv *second,
                       __global const IndexPair_t *out, const unsigned N)
{
    int tIdx = get_group_id(0) * get_local_size(0) * copyPairIter + get_local_id(0);
    const int blockDimX = get_local_size(0);

    for(int i = tIdx; i < N; i += blockDimX) {
        first[i]  = out[i].first;
        second[i] = out[i].second;
    }
}
