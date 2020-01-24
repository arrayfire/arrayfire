/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/kernel_cache.hpp>

#include <cl2hpp.hpp>
#include <common/defines.hpp>
#include <platform.hpp>
#include <program.hpp>

using std::string;
using std::vector;

namespace common {

detail::DevPtrType Kernel::get(const char *name) {
    // perhaps throw if code reaches here
    return nullptr;
}

void Kernel::copyToReadOnly(detail::DevPtrType dst, detail::DevPtrType src,
                            size_t bytes) {
    opencl::getQueue().enqueueCopyBuffer(*src, *dst, 0, 0, bytes);
}

template<typename T>
void Kernel::setScalar(detail::DevPtrType dst, T value) {
    // CU_CHECK(cuMemcpyHtoDAsync(dst, &value, sizeof(T),
    // cuda::getActiveStream()));
    // CU_CHECK(cuStreamSynchronize(cuda::getActiveStream()));
}

template<typename T>
void Kernel::getScalar(T &out, detail::DevPtrType src) {
    // CU_CHECK(cuMemcpyDtoHAsync(&out, src, sizeof(T),
    // cuda::getActiveStream()));
    // CU_CHECK(cuStreamSynchronize(cuda::getActiveStream()));
}

template void Kernel::setScalar<int>(detail::DevPtrType, int);
template void Kernel::getScalar<int>(int &, detail::DevPtrType);

void compileKernel(Kernel &out, const string &kernelName,
                   const string &tInstance, const vector<string> &sources,
                   const vector<string> &compileOpts, const bool isJIT) {
    UNUSED(isJIT);
    UNUSED(tInstance);

    auto prog = detail::buildProgram(sources, compileOpts);
    out.prog  = new cl::Program(prog);
    out.kern  = new cl::Kernel(*static_cast<cl::Program *>(out.prog),
                              kernelName.c_str());
}

}  // namespace common
