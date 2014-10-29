#include <af/index.h>
#include "error.hpp"

namespace af
{

array moddims(const array& in, const unsigned ndims, const dim_type * const dims)
{
    af_array out = 0;
    AF_THROW(af_moddims(&out, in.get(), ndims, dims));
    return array(out);
}

array tile(const array& in, const unsigned x, const unsigned y, const unsigned z, const unsigned w)
{
    af_array out = 0;
    AF_THROW(af_tile(&out, in.get(), x, y, z, w));
    return array(out);
}

array reorder(const array& in, const unsigned x, const unsigned y, const unsigned z, const unsigned w)
{
    af_array out = 0;
    AF_THROW(af_reorder(&out, in.get(), x, y, z, w));
    return array(out);
}

array shift(const array& in, const int x, const int y, const int z, const int w)
{
    af_array out = 0;
    AF_THROW(af_shift(&out, in.get(), x, y, z, w));
    return array(out);
}

}
