<a href="http://arrayfire.com/"><img src="http://arrayfire.com/logos/arrayfire_logo_whitebkgnd.png" width="300"></a>

ArrayFire is a general-purpose library that simplifies the process of developing
software that targets parallel and massively-parallel architectures including
CPUs, GPUs, and other hardware acceleration devices.

Several of ArrayFire's benefits include:

* [Easy to use](http://arrayfire.org/docs/gettingstarted.htm), stable,
  [well-documented](http://arrayfire.org/docs) API
* Rigorously tested for performance and accuracy
* Commercially friendly open-source licensing
* Commercial support from [ArrayFire](http://arrayfire.com)
* [Read about more benefits on arrayfire.com](http://arrayfire.com/the-arrayfire-library/)

ArrayFire provides software developers with a high-level
abstraction of data which resides on the accelerator, the `af::array` object.
Developers write code which performs operations on ArrayFire arrays which, in turn,
are automatically translated into near-optimal kernels that execute on the computational
device.

ArrayFire is successfully used on devices ranging from low-power mobile phones
to high-power GPU-enabled supercomputers. ArrayFire runs on CPUs from all
major vendors (Intel, AMD, ARM), GPUs from the prominent manufacturers
(NVIDIA, AMD, and Qualcomm), as well as a variety of other accelerator devices
on Windows, Mac, and Linux.

## Installation

You can install the ArrayFire library from one of the following ways:

#### Official installers

Execute one of our [official binary installers](https://arrayfire.com/download)
for Linux, OSX, and Windows platforms.

#### Build from source

Build from source by following instructions on our
[wiki](https://github.com/arrayfire/arrayfire/wiki).

## Examples

The following examples are simplified versions of
[`helloworld.cpp`](https://github.com/arrayfire/arrayfire/tree/devel/examples/helloworld/helloworld.cpp)
and
[`conway_pretty.cpp`](https://github.com/arrayfire/arrayfire/tree/devel/examples/graphics/conway_pretty.cpp),
respectively. For more code examples, visit the
[`examples/`](https://github.com/arrayfire/arrayfire/tree/devel/examples)
directory.

#### Hello, world!

```cpp
array A = randu(5, 3, f32); // Create 5x3 matrix of random floats on the GPU
array B = sin(A) + 1.5;     // Element-wise arithmetic
array C = fft(B);           // Fourier transform the result

float d[] = { 1, 2, 3, 4, 5, 6 };
array D(2, 3, d, afHost);   // Create 2x3 matrix from host data
D.col(0) = D.col(end);      // Copy last column onto first

array vals, inds;
sort(vals, inds, A);        // Sort A and print sorted array and corresponding indices
af_print(vals);
af_print(inds);
```

#### Conway's Game of Life

Visit the
[Wikipedia page](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) for a
description of Conway's Game of Life.

```cpp
static const float h_kernel[] = {1, 1, 1, 1, 0, 1, 1, 1, 1};
static const array kernel(3, 3, h_kernel, afHost);

array state = (randu(128, 128, f32) > 0.5).as(f32); // Generate starting state
Window myWindow(256, 256);
while(!myWindow.close()) {
  array nHood = convolve(state, kernel); // Obtain neighbors
  array C0 = (nHood == 2);               // Generate conditions for life
  array C1 = (nHood == 3);
  state = state * C0 + C1;               // Update state
  myWindow.image(state);                 // Display
}

```

<p align="center">
<img src="https://github.com/arrayfire/assets/blob/master/gifs/conway.gif" alt="Conway's Game of Life" height="256" width="256">
</p>

## Documentation

You can find our complete documentation [here](http://www.arrayfire.com/docs/index.htm).

Quick links:

* [List of functions](http://www.arrayfire.org/docs/group__arrayfire__func.htm)
* [Tutorials](http://www.arrayfire.org/docs/usergroup0.htm)
* [Examples](http://www.arrayfire.org/docs/examples.htm)
* [Blog](http://arrayfire.com/blog/)

## Language wrappers

ArrayFire has several official and third-party language wrappers.

__Official wrappers__

We currently support the following language wrappers for ArrayFire:

* [`arrayfire-python`](https://github.com/arrayfire/arrayfire-python)
* [`arrayfire-rust`](https://github.com/arrayfire/arrayfire-rust)

Wrappers for other languages are a work-in-progress:
  [.NET](https://github.com/arrayfire/arrayfire-dotnet),
  [Fortran](https://github.com/arrayfire/arrayfire-fortran),
  [Go](https://github.com/arrayfire/arrayfire-go),
  [Java](https://github.com/arrayfire/arrayfire-java),
  [Lua](https://github.com/arrayfire/arrayfire-lua),
  [NodeJS](https://github.com/arrayfire/arrayfire-js),
  [R](https://github.com/arrayfire/arrayfire-r),
  [Ruby](https://github.com/arrayfire/arrayfire-rb)

__Third-party wrappers__

The following wrappers are being maintained and supported by third parties:

* [`ArrayFire.jl`](https://github.com/JuliaComputing/ArrayFire.jl)
* [`ArrayFire-Nim`](https://github.com/bitstormGER/ArrayFire-Nim)

## Contributing

Contributions of any kind are welcome! Please refer to
[CONTRIBUTING.md](https://github.com/arrayfire/arrayfire/blob/master/CONTRIBUTING.md)
to learn more about how you can get involved with ArrayFire.

## Citations and Acknowledgements

If you redistribute ArrayFire, please follow the terms established in
[the license](LICENSE). If you wish to cite ArrayFire in an academic
publication, please use the following [citation document](.github/CITATION.md).

ArrayFire development is funded by ArrayFire LLC and several third parties,
please see the list of [acknowledgements](ACKNOWLEDGEMENTS.md) for further
details.

## Support and Contact Info

* [Slack Chat](https://join.slack.com/t/arrayfire-org/shared_invite/MjI4MjIzMDMzMTczLTE1MDI5ODg4NzYtN2QwNGE3ODA5OQ)
* [Google Groups](https://groups.google.com/forum/#!forum/arrayfire-users)
* ArrayFire Services:  [Consulting](http://arrayfire.com/consulting/)  |  [Support](http://arrayfire.com/support/)   |  [Training](http://arrayfire.com/training/)

## Trademark Policy

The literal mark “ArrayFire” and ArrayFire logos are trademarks of
AccelerEyes LLC DBA ArrayFire.
If you wish to use either of these marks in your own project, please consult
[ArrayFire's Trademark Policy](http://arrayfire.com/trademark-policy/)
