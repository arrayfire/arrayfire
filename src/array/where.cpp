#include <af/array.h>
#include <af/data.h>
#include "error.hpp"

namespace af
{
    array where(const array& in)
    {
        af_array out = 0;
        AF_THROW(af_where(&out, in.get()));
        return array(out);
    }
}
