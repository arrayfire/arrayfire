#include <af/image.h>
#include "error.hpp"

namespace af
{

array regions(const array& in, af_connectivity_type connectivity, af_dtype type)
{
    af_array temp = 0;
    AF_THROW(af_regions(&temp, in.get(), connectivity, type));
    return array(temp);
}

}
