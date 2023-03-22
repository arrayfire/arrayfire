
<p align="center"><a href="http://arrayfire.com/"><img src="http://arrayfire.com/logos/arrayfire_logo_whitebkgnd.png" width="800"></a></p>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bOQY_XRn7JWWGRU6tDb5p2KMSRsYrzUE?usp=sharing)

ArrayFire is a general-purpose tensor library that simplifies the process of
software development for the parallel architectures found in CPUs, GPUs, and
other hardware acceleration devices. The library serves users in every technical
computing market.

Several of ArrayFire's benefits include:

* Hundreds of accelerated [tensor computing functions](https://arrayfire.org/docs/group__arrayfire__func.htm), in the following areas:
    * Array handling
    * Computer vision
    * Image processing
    * Linear algebra
    * Machine learning
    * Standard math
    * Signal Processing
    * Statistics
    * Vector algorithms
* [Easy to use](http://arrayfire.org/docs/gettingstarted.htm), stable,
  [well-documented](http://arrayfire.org/docs) API
* Rigorous benchmarks and tests ensuring top performance and numerical accuracy
* Cross-platform compatibility with support for CUDA, OpenCL, and native CPU on Windows, Mac, and Linux
* Built-in visualization functions through [Forge](https://github.com/arrayfire/forge)
* Commercially friendly open-source licensing
* Enterprise support from [ArrayFire](http://arrayfire.com)

ArrayFire provides software developers with a high-level abstraction of data
that resides on the accelerator, the `af::array` object. Developers write code
that performs operations on ArrayFire arrays, which, in turn, are automatically
translated into near-optimal kernels that execute on the computational device.

ArrayFire runs on devices ranging from low-power mobile phones to high-power
GPU-enabled supercomputers. ArrayFire runs on CPUs from all major vendors
(Intel, AMD, ARM), GPUs from the prominent manufacturers (NVIDIA, AMD, and
Qualcomm), as well as a variety of other accelerator devices on Windows, Mac,
and Linux.

# Getting ArrayFire

Instructions to [install][32] or to build ArrayFire from source can be found on the [wiki][1].

### Conway's Game of Life Using ArrayFire

Visit the [Wikipedia page][2] for a description of Conway's Game of Life.

<img align="left" src="https://github.com/arrayfire/assets/blob/master/gifs/conway.gif" alt="Conway's Game of Life" height="256" width="256">

```cpp
static const float h_kernel[] = { 1, 1, 1, 1, 0, 1, 1, 1, 1 };
static const array kernel(3, 3, h_kernel, afHost);

array state = (randu(128, 128, f32) > 0.5).as(f32); // Init state
Window myWindow(256, 256);
while(!myWindow.close()) {
    array nHood = convolve(state, kernel); // Obtain neighbors
    array C0 = (nHood == 2);  // Generate conditions for life
    array C1 = (nHood == 3);
    state = state * C0 + C1;  // Update state
    myWindow.image(state);    // Display
}
```
The complete source code can be found [here][3].

### Perceptron

<img align="left" src="https://github.com/arrayfire/assets/blob/imgs_readme_improv/gifs/perceptron.gif" alt="Perceptron" height="400" width="300">

```cpp
array predict(const array &X, const array &W) {
    return sigmoid(matmul(X, W));
}

array train(const array &X, const array &Y,
        double alpha = 0.1, double maxerr = 0.05,
        int maxiter = 1000, bool verbose = false) {
    array Weights = constant(0, X.dims(1), Y.dims(1));

    for (int i = 0; i < maxiter; i++) {
        array P   = predict(X, Weights);
        array err = Y - P;
        if (mean<float>(abs(err) < maxerr) break;
        Weights += alpha * matmulTN(X, err);
    }
    return Weights;
}
...

array Weights = train(train_feats, train_targets);
array test_outputs  = predict(test_feats, Weights);
display_results<true>(test_images, test_outputs,
                      test_targets, 20);
```

The complete source code can be found [here][31].

For more code examples, visit the [`examples/`][4] directory.

# Documentation

You can find the complete documentation [here](http://www.arrayfire.com/docs/index.htm).

Quick links:

* [List of functions](http://www.arrayfire.org/docs/group__arrayfire__func.htm)
* [Tutorials](http://arrayfire.org/docs/tutorials.htm)
* [Examples](http://www.arrayfire.org/docs/examples.htm)
* [Blog](http://arrayfire.com/blog/)

# Language support

ArrayFire has several official and community maintained language API's:

[![C++][5]][6] [![Python][7]][8] [![Rust][9]][10] [![Julia][27]][28]<sub><span>&#8224;</span></sub>
[![Nim][29]][30]<sub><span>&#8224;</span></sub>

<sup><span>&#8224;</span></sup>&nbsp; Community maintained wrappers

__In-Progress Wrappers__

[![.NET][11]][12] [![Fortran][13]][14] [![Go][15]][16]
[![Java][17]][18] [![Lua][19]][20] [![NodeJS][21]][22] [![R][23]][24] [![Ruby][25]][26]

# Contributing

The community of ArrayFire developers invites you to build with us if you are
interested and able to write top-performing tensor functions. Together we can
fulfill [The ArrayFire
Mission](https://github.com/arrayfire/arrayfire/wiki/The-ArrayFire-Mission-Statement)
for fast scientific computing for all.

Contributions of any kind are welcome! Please refer to [the
wiki](https://github.com/arrayfire/arrayfire/wiki) and our [Code of Conduct](33)
to learn more about how you can get involved with the ArrayFire Community
through [Sponsorship](https://github.com/arrayfire/arrayfire/wiki/Sponsorship),
[Developer
Commits](https://github.com/arrayfire/arrayfire/wiki/Contributing-Code-to-ArrayFire),
or [Governance](https://github.com/arrayfire/arrayfire/wiki/Governance).

# Citations and Acknowledgements

If you redistribute ArrayFire, please follow the terms established in [the
license](LICENSE). If you wish to cite ArrayFire in an academic publication,
please use the following [citation document](.github/CITATION.md).

ArrayFire development is funded by AccelerEyes LLC and several third parties,
please see the list of [acknowledgements](ACKNOWLEDGEMENTS.md) for an expression
of our gratitude.

# Support and Contact Info

* [Slack Chat](https://join.slack.com/t/arrayfire-org/shared_invite/MjI4MjIzMDMzMTczLTE1MDI5ODg4NzYtN2QwNGE3ODA5OQ)
* [Google Groups](https://groups.google.com/forum/#!forum/arrayfire-users)
* ArrayFire Services:  [Consulting](http://arrayfire.com/consulting)  |  [Support](http://arrayfire.com/download)   |  [Training](http://arrayfire.com/training)

# Trademark Policy

The literal mark "ArrayFire" and ArrayFire logos are trademarks of
AccelerEyes LLC (dba ArrayFire).
If you wish to use either of these marks in your own project, please consult
[ArrayFire's Trademark Policy](http://arrayfire.com/trademark-policy/)

[1]: https://github.com/arrayfire/arrayfire/wiki
[2]: https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
[3]: https://github.com/arrayfire/arrayfire/blob/master/examples/graphics/conway_pretty.cpp
[4]: https://github.com/arrayfire/arrayfire/blob/master/examples/
[5]: https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white
[6]: http://arrayfire.org/docs/gettingstarted.htm#gettingstarted_api_usage
[7]: https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white
[8]: https://github.com/arrayfire/arrayfire-python
[9]: https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white
[10]: https://github.com/arrayfire/arrayfire-rust
[11]: https://img.shields.io/badge/.NET-5C2D91?style=for-the-badge&logo=.net&logoColor=white
[12]: https://github.com/arrayfire/arrayfire-dotnet
[13]: https://img.shields.io/badge/F-Fortran-734f96?style=for-the-badge
[14]: https://github.com/arrayfire/arrayfire-fortran
[15]: https://img.shields.io/badge/go-%2300ADD8.svg?style=for-the-badge&logo=go&logoColor=white
[16]: https://github.com/arrayfire/arrayfire-go
[17]: https://img.shields.io/badge/java-%23ED8B00.svg?style=for-the-badge&logo=java&logoColor=white
[18]: https://github.com/arrayfire/arrayfire-java
[19]: https://img.shields.io/badge/lua-%232C2D72.svg?style=for-the-badge&logo=lua&logoColor=white
[20]: https://github.com/arrayfire/arrayfire-lua
[21]: https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E
[22]: https://github.com/arrayfire/arrayfire-js
[23]: https://img.shields.io/badge/r-%23276DC3.svg?style=for-the-badge&logo=r&logoColor=white
[24]: https://github.com/arrayfire/arrayfire-r
[25]: https://img.shields.io/badge/ruby-%23CC342D.svg?style=for-the-badge&logo=ruby&logoColor=white
[26]: https://github.com/arrayfire/arrayfire-rb
[27]: https://img.shields.io/badge/j-Julia-cb3c33?style=for-the-badge&labelColor=4063d8
[28]: https://github.com/JuliaComputing/ArrayFire.jl
[29]: https://img.shields.io/badge/n-Nim-000000?style=for-the-badge&labelColor=efc743
[30]: https://github.com/bitstormGER/ArrayFire-Nim
[31]: https://github.com/arrayfire/arrayfire/blob/master/examples/machine_learning/perceptron.cpp
[32]: https://github.com/arrayfire/arrayfire/wiki/Getting-ArrayFire
[33]: https://github.com/arrayfire/arrayfire/wiki/Code-Of-Conduct
