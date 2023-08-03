#ifndef __cplusplus
#  error "A C compiler has been selected for C++."
#endif

#include "CMakeCompilerABI.h"
#include <sycl/sycl.hpp>

int main(int argc, char* argv[])
{
  int require = 0;
  require += info_sizeof_dptr[argc];
  require += info_byte_order_big_endian[argc];
  require += info_byte_order_little_endian[argc];
#if defined(ABI_ID)
  require += info_abi[argc];
#endif
  static_cast<void>(argv);

  int count = 0;
  auto platforms = sycl::platform::get_platforms();
  for(sycl::platform &platform : platforms) {
    count += platform.get_devices().size();
  }

  if(count == 0) {
    std::fprintf(stderr, "No SYCL devices found.\n");
    return -1;
  }

  return require;
}
