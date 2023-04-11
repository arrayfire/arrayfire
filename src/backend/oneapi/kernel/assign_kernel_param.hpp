
#include <sycl/sycl.hpp>

#include <array>

#pragma once

namespace arrayfire {
namespace oneapi {

typedef struct {
    int offs[4];
    int strds[4];
    bool isSeq[4];
    std::array<sycl::accessor<unsigned int, 1, sycl::access::mode::read,
                              sycl::access::target::device>,
               4>
        ptr;

} AssignKernelParam;

using IndexKernelParam = AssignKernelParam;

}  // namespace oneapi
}  // namespace arrayfire
