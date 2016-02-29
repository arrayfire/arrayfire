<a href="http://arrayfire.com/"><img src="http://arrayfire.com/logos/arrayfire_logo_whitebkgnd.png" width="300"></a>

|         | Linux x86 | Linux armv7l | Linux aarch64 | Windows | OSX |
|:-------:|:---------:|:------------:|:-------------:|:-------:|:---:|
| Build   | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-linux/build/devel)](http://ci.arrayfire.org/job/arrayfire-linux/job/build/branch/devel/) | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-tegrak1/build/devel)](http://ci.arrayfire.org/job/arrayfire-tegrak1/job/build/branch/devel/) | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-tegrax1/build/devel)](http://ci.arrayfire.org/job/arrayfire-tegrax1/job/build/branch/devel/) | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-windows/build/devel)](http://ci.arrayfire.org/job/arrayfire-windows/job/build/branch/devel/) | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-osx/build/devel)](http://ci.arrayfire.org/job/arrayfire-osx/job/build/branch/devel/) |
| Test    | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-linux/test/devel)](http://ci.arrayfire.org/job/arrayfire-linux/job/test/branch/devel/) | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-tegrak1/test/devel)](http://ci.arrayfire.org/job/arrayfire-tegrak1/job/test/branch/devel/) | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-tegrax1/test/devel)](http://ci.arrayfire.org/job/arrayfire-tegrax1/job/test/branch/devel/) | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-windows/test/devel)](http://ci.arrayfire.org/job/arrayfire-windows/job/test/branch/devel/) | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-osx/test/devel)](http://ci.arrayfire.org/job/arrayfire-osx/job/test/branch/devel/) |

ArrayFire is a high performance software library for parallel computing with an
easy-to-use API. Its **array** based function set makes parallel programming
simple.

ArrayFire's multiple backends (**CUDA**, **OpenCL** and native **CPU**) make it
platform independent and highly portable.

A few lines of code in ArrayFire can replace dozens of lines of parallel
computing code, saving you valuable time and lowering development costs.

### Example

``` C++

#include <arrayfire.h>
#include <cstdio>

using namespace af;

int main(int argc, char *argv[])
{
  try {

    // Select a device and display arrayfire info
    int device = argc > 1 ? atoi(argv[1]) : 0;
    af::setDevice(device);
    af::info();

    printf("Create a 5-by-3 matrix of random floats on the GPU\n");
    array A = randu(5,3, f32);
    af_print(A);

    printf("Element-wise arithmetic\n");
    array B = sin(A) + 1.5;
    af_print(B);

    printf("Negate the first three elements of second column\n");
    B(seq(0, 2), 1) = B(seq(0, 2), 1) * -1;
    af_print(B);

    printf("Fourier transform the result\n");
    array C = fft(B);
    af_print(C);

    printf("Grab last row\n");
    array c = C.row(end);
    af_print(c);

    printf("Create 2-by-3 matrix from host data\n");
    float d[] = { 1, 2, 3, 4, 5, 6 };
    array D(2, 3, d, af::afHost);
    af_print(D);

    printf("Copy last column onto first\n");
    D.col(0) = D.col(end);
    af_print(D);

    // Sort A
    printf("Sort A and print sorted array and corresponding indices\n");
    array vals, inds;
    sort(vals, inds, A);
    af_print(vals);
    af_print(inds);

  } catch (af::exception& e) {
    fprintf(stderr, "%s\n", e.what());
    throw;
  }
}

```

Visit the [examples](https://github.com/arrayfire/arrayfire/tree/devel/examples)
directory for more examples.

### Installing

Visit our [downloads](http://go.arrayfire.com/l/37882/2015-03-31/mmhqy) page to
download ArrayFire binary installers.

### Building

Please follow the instructions on our
[wiki](https://github.com/arrayfire/arrayfire/wiki) to build ArrayFire from
source.

### Documentation

You can find our complete [documentation](http://www.arrayfire.com/docs/index.htm) and [tutorials](http://www.arrayfire.com/docs/usergroup0.htm) on the
[ArrayFire website](http://www.arrayfire.com).

### Contributing

Contributions of any kind are welcome! Please refer to
[CONTRIBUTING.md](https://github.com/arrayfire/arrayfire/blob/master/CONTRIBUTING.md)
to learn more about how you can get involved with ArrayFire.

### Support and Contact Info [![Join the chat at https://gitter.im/arrayfire/arrayfire](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/arrayfire/arrayfire?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

* [Mailing list](https://groups.google.com/forum/#!forum/arrayfire-users)
* [Blog](http://arrayfire.com/blog/)
* ArrayFire Services:  [Consulting](http://arrayfire.com/consulting/)  |  [Support](http://arrayfire.com/support/)   |  [Training](http://arrayfire.com/training/)
