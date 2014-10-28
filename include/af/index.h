#pragma once
#include <af/array.h>
#include <af/defines.h>

#if __cplusplus
namespace af
{

AFAPI array tile(const array& in, const unsigned x, const unsigned y=1, const unsigned z=1, const unsigned w=1);

AFAPI array reorder(const array& in, const unsigned x, const unsigned y=1, const unsigned z=2, const unsigned w=3);

AFAPI array shift(const array& in, const int x, const int y=0, const int z=0, const int w=0);

AFAPI array moddims(const array& in, const unsigned ndims, const dim_type * const dims);

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    // Create a new af_array by indexing from existing af_array.
    // This takes the form `out = in(seq_a, seq_b)`
    AFAPI af_err af_index(af_array *out, const af_array in, unsigned ndims, const af_seq* const index);

    // Tile an Array
    AFAPI af_err af_tile(af_array *out, const af_array in,
                         const unsigned x, const unsigned y, const unsigned z, const unsigned w);

    // Reorder an Array
    AFAPI af_err af_reorder(af_array *out, const af_array in,
                            const unsigned x, const unsigned y, const unsigned z, const unsigned w);

    // Reorder an Array
    AFAPI af_err af_shift(af_array *out, const af_array in, const int x, const int y, const int z, const int w);

    // re-shape the the dimensions of the input array
    AFAPI af_err af_moddims(af_array *out, const af_array in, const unsigned ndims, const dim_type * const dims);

#ifdef __cplusplus
}
#endif
