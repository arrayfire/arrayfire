#include <af/blas.h>
#include "error.hpp"

namespace af
{

array transpose(const array& in)
{
    af_array out = 0;
    AF_THROW(af_transpose(&out, in.get()));
    return array(out);
}

}
