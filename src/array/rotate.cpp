#include <af/image.h>
#include "error.hpp"

namespace af
{

array rotate(const array& in, const float theta, const bool crop, const bool recenter)
{
    af_array out = 0;
    AF_THROW(af_rotate(&out, in.get(), theta, crop, recenter));
    return array(out);
}

}
