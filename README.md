<a href="http://arrayfire.com/"><img src="http://arrayfire.com/logos/arrayfire_logo_whitebkgnd.png" width="300"></a>

ArrayFire is a high performance software library for parallel computing with an easy-to-use API. Its **array** based function set makes parallel programming simple.

ArrayFire's multiple backends (**CUDA**, **OpenCL** and native **CPU**) make it platform independent and highly portable.

A few lines of code in ArrayFire can replace dozens of lines of parallel computing code, saving you valuable time and lowering development costs.

### Build Status
|                 | Build           | Tests           |
|-----------------|-----------------|-----------------|
| Linux x86       | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-linux/devel)](http://ci.arrayfire.org/job/arrayfire-linux/branch/devel/)      | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-linux-test/devel)](http://ci.arrayfire.org/job/arrayfire-linux-test/branch/devel/)
| Linux Tegra     | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-tegra/devel)](http://ci.arrayfire.org/job/arrayfire-tegra/branch/devel/)      | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-tegra-test/devel)](http://ci.arrayfire.org/job/arrayfire-tegra-test/branch/devel/)
| Windows         | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-windows/devel)](http://ci.arrayfire.org/job/arrayfire-windows/branch/devel/)  | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-windows-test/devel)](http://ci.arrayfire.org/job/arrayfire-windows-test/branch/devel/)                     |
| OSX             | [![Build Status](http://ci.arrayfire.org/buildStatus/icon?job=arrayfire-osx/devel)](http://ci.arrayfire.org/job/arrayfire-osx/branch/devel/)          |                 |

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

        // Create a 5-by-3 matrix of random floats on the GPU
        array A = randu(5,3, f32);

        // Element-wise arithmetic
        array B = sin(A) + 1.5;

        // Negate the first three elements of second column
        B(seq(0, 2), 1) = B(seq(0, 2), 1) * -1;

        // Fourier transform the result
        array C = fft(B);

        // Grab last row
        array c = C.row(end);

        // Create 2-by-3 matrix from host data
        float d[] = { 1, 2, 3, 4, 5, 6 };
        array D(2, 3, d, af::afHost);

        // Copy last column onto first
        D.col(0) = D.col(end);
        
        // Sort A and print sorted array and corresponding indices
        array vals, inds;
        sort(vals, inds, A);

    } catch (const af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }
}

```

### Documentation

You can find our complete documentation [here](http://www.arrayfire.com/docs/index.htm).

Quick links:

- [List of functions](http://www.arrayfire.com/docs/group__arrayfire__func.htm)
- [Tutorials](http://www.arrayfire.com/docs/gettingstarted.htm)
- [Examples](http://www.arrayfire.com/docs/examples.htm)

### Build ArrayFire from source

To build ArrayFire from source, please follow the instructions on our [wiki](https://github.com/arrayfire/arrayfire/wiki).

### Download ArrayFire Installers

We are currently working on bring out installers for the open source version. Please try to build this using our [wiki](https://github.com/arrayfire/arrayfire/wiki) page.

Installers for the older (commercial) versions of ArrayFire can be freely downloaded from [here](https://arrayfire.com/download). This will require licensing.

### Contribute

Contributions of any kind are welcome! Please refer to [this document](https://github.com/arrayfire/arrayfire/blob/master/CONTRIBUTING.md) to learn more about how you can get involved with ArrayFire.

### Contact us

* Google Groups: https://groups.google.com/forum/#!forum/arrayfire-users
* ArrayFire Forums: http://arrayfire.com/forums/
