#include <af/index.h>
#include "error.hpp"

namespace af
{

AFAPI array moddims(const array& in, const unsigned ndims, const dim_type * const dims)
{
    af_array out = 0;
    AF_THROW(af_moddims(&out, in.get(), ndims, dims));
    return array(out);
}

}
