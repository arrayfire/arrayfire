/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/index.h>
// The following should be included using double quotes
// to enable it's use in unified wrapper
#include "err_common.hpp"

af_seq af_make_seq(double begin, double end, double step)
{
    af_seq seq = {begin, end, step};
    return seq;
}

af_err af_create_indexers(af_index_t** indexers)
{
    try {
        af_index_t* out = new af_index_t[4];
        std::swap(*indexers, out);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_set_array_indexer(af_index_t* indexer, const af_array idx, const dim_t dim)
{
    try {
        ARG_ASSERT(0, (indexer!=NULL));
        ARG_ASSERT(1, (idx!=NULL));
        ARG_ASSERT(2, (dim>=0 && dim<=3));
        indexer[dim].idx.arr = idx;
        indexer[dim].isBatch = false;
        indexer[dim].isSeq   = false;
    }
    CATCHALL
    return AF_SUCCESS;
}

af_err af_set_seq_indexer(af_index_t* indexer, const af_seq* idx, const dim_t dim, const bool is_batch)
{
    try {
        ARG_ASSERT(0, (indexer!=NULL));
        ARG_ASSERT(1, (idx!=NULL));
        ARG_ASSERT(2, (dim>=0 && dim<=3));
        indexer[dim].idx.seq = *idx;
        indexer[dim].isBatch = is_batch;
        indexer[dim].isSeq   = true;
    }
    CATCHALL
    return AF_SUCCESS;
}

af_err af_set_seq_param_indexer(af_index_t* indexer,
                              const double begin, const double end, const double step,
                              const dim_t dim, const bool is_batch)
{
    try {
        ARG_ASSERT(0, (indexer!=NULL));
        ARG_ASSERT(4, (dim>=0 && dim<=3));
        indexer[dim].idx.seq = af_make_seq(begin, end, step);
        indexer[dim].isBatch = is_batch;
        indexer[dim].isSeq   = true;
    }
    CATCHALL
    return AF_SUCCESS;
}

af_err af_release_indexers(af_index_t* indexers)
{
    try {
        delete[] indexers;
    }
    CATCHALL;
    return AF_SUCCESS;
}
