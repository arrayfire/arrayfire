<a href="http://arrayfire.com/"><img src="http://arrayfire.com/logos/arrayfire_logo_whitebkgnd.png" width="300"></a>

ArrayFire is a high performance software library for parallel computing with an easy-to-use API. Its **array** based function set makes parallel programming simple.

ArrayFire's multiple backends (**CUDA**, **OpenCL** and native **CPU**) make it platform independent and highly portable.

A few lines of code in ArrayFire can replace dozens of lines of parallel computing code, saving you valuable time and lowering development costs.

### Build ArrayFire from source
To build ArrayFire from source, please follow the instructions on our [wiki](https://github.com/arrayfire/arrayfire/wiki).

### Download ArrayFire Installers
ArrayFire binary installers can be downloaded at the [ArrayFire Downloads](http://go.arrayfire.com/l/37882/2015-03-31/mmhqy) page.

### Support and Contact Info [![Join the chat at https://gitter.im/arrayfire/arrayfire](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/arrayfire/arrayfire?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

* Google Groups: https://groups.google.com/forum/#!forum/arrayfire-users
* ArrayFire Services:  [Consulting](http://arrayfire.com/consulting/)  |  [Support](http://arrayfire.com/support/)   |  [Training](http://arrayfire.com/training/)
* ArrayFire Blogs: http://arrayfire.com/blog/
* Email: <mailto:technical@arrayfire.com>

### Build Status
|         | Linux x86 | Linux armv7l | Linux aarch64 | Windows | OSX |
|:-------:|:---------:|:------------:|:-------------:|:-------:|:---:|
| Build   | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-linux/devel)](http://ci.arrayfire.org/job/arrayfire-linux/branch/devel/) | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-tegrak1/devel)](http://ci.arrayfire.org/job/arrayfire-tegrak1/branch/devel/) | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-tegrax1/devel)](http://ci.arrayfire.org/job/arrayfire-tegrax1/branch/devel/) | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-windows/devel)](http://ci.arrayfire.org/job/arrayfire-windows/branch/devel/) | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-osx/devel)](http://ci.arrayfire.org/job/arrayfire-osx/branch/devel/) |
| Test    | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-linux-test/devel)](http://ci.arrayfire.org/job/arrayfire-linux-test/branch/devel/) | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-tegrak1-test/devel)](http://ci.arrayfire.org/job/arrayfire-tegrak1-test/branch/devel/) | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-tegrax1-test/devel)](http://ci.arrayfire.org/job/arrayfire-tegrax1-test/branch/devel/) | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-windows-test/devel)](http://ci.arrayfire.org/job/arrayfire-windows-test/branch/devel/) | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-osx-test/devel)](http://ci.arrayfire.org/job/arrayfire-osx-test/branch/devel/) |

Test coverage: [![Coverage Status](https://coveralls.io/repos/arrayfire/arrayfire/badge.svg?branch=HEAD)](https://coveralls.io/r/arrayfire/arrayfire?branch=HEAD)

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

### Documentation

You can find our complete documentation over [here](http://www.arrayfire.com/docs/index.htm).

Quick links:

- [Download Binaries](http://www.arrayfire.com/download/)
- [List of functions](http://www.arrayfire.com/docs/group__arrayfire__func.htm)
- [Tutorials](http://www.arrayfire.com/docs/gettingstarted.htm)
- [Examples](http://www.arrayfire.com/docs/examples.htm)

### Contribute

Contributions of any kind are welcome! Please refer to
[this document](https://github.com/arrayfire/arrayfire/blob/master/CONTRIBUTING.md)
 to learn more about how you can get involved with ArrayFire.

## Citations and Acknowledgements

If you redistribute ArrayFire, please follow the terms established in
[the license](LICENSE).
If you wish to cite ArrayFire in an academic publication, please use the
following reference:

Formatted:
```
Yalamanchili, P., Arshad, U., Mohammed, Z., Garigipati, P., Entschev, P.,
Kloppenborg, B., Malcolm, J. and Melonakos, J. (2015).
ArrayFire - A high performance software library for parallel computing with an
easy-to-use API. Atlanta: AccelerEyes. Retrieved from https://github.com/arrayfire/arrayfire
```

BibTeX:
```bibtex
@misc{Yalamanchili2015,
abstract = {ArrayFire is a high performance software library for parallel computing with an easy-to-use API. Its array based function set makes parallel programming simple. ArrayFire's multiple backends (CUDA, OpenCL and native CPU) make it platform independent and highly portable. A few lines of code in ArrayFire can replace dozens of lines of parallel computing code, saving you valuable time and lowering development costs.},
address = {Atlanta},
author = {Yalamanchili, Pavan and Arshad, Umar and Mohammed, Zakiuddin and Garigipati, Pradeep and Entschev, Peter and Kloppenborg, Brian and Malcolm, James and Melonakos, John},
publisher = {AccelerEyes},
title = {{ArrayFire - A high performance software library for parallel computing with an easy-to-use API}},
url = {https://github.com/arrayfire/arrayfire},
year = {2015}
}
```

ArrayFire development is funded by ArrayFire LLC and several third parties,
please see the list of [acknowledgements](https://github.com/arrayfire/arrayfire/blob/master/ACKNOWLEDGEMENTS.md) for further details.

