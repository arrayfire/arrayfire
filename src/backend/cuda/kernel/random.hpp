#include <af/defines.h>
#include <backend.hpp>
#include "../helper.hpp"

namespace cuda
{
namespace kernel
{

    template<typename T>
    void randu(T *out, size_t elements);

    template<typename T>
    void randn(T *out, size_t elements);

}
}
