ArrayFire example code build instructions
=====

The source code contained within this directory can be compiled within the 
ArrayFire source tree or as part of a stand-alone project. 

## Building examples while compiling ArrayFire

By default, ArrayFire will build the examples as part of the standard build
process; however, the compiled examples are not packaged in the ArrayFire
installer. After compiling ArrayFire, the examples will be in subdirectories
located in the `build/examples` directory.

If you wish to disable example compilation, simply set the `BUILD_EXAMPLES`
variable to `OFF` in the CMake GUI or `ccmake` curses wrapper. If you are
using the command-line version of `cmake`, simply specify 
`-DBUILD_EXAMPLES=OFF` as an argument.

## Building examples as a stand-alone project

Once ArrayFire is installed on your machine, the contents of this directory
will be copied to a documentation directory on your computer. For example,
on Linux this will be `/usr/share/doc/arrayfire`.

To compile the examples, simply copy the ArrayFire example directory to
a directory in which you have write permissions, and compile the examples
using `cmake` and `make`:

    cp -r /usr/share/doc/arrayfire/examples ~/arrayfire_examples
    cd ~/arrayfire_examples
    mkdir build
    cd build
    cmake ..
    make
    
If ArrayFire is not installed to a system directory, you will need to specify
the directory which contains the `ArrayFireConfig.cmake` as an argument to the
`cmake` invocation. This configuration file is located within the 
`share/ArrayFire` subdirectory of the ArrayFire installation. For example,
if you were to install ArrayFire to the `local` directory within your home
folder, the invocation of `cmake` above would be replaced with the following:

    cmake -DArrayFire_ROOT=~/local/share/ArrayFire/ ..
    
### Support and Contact Info

* Google Groups: https://groups.google.com/forum/#!forum/arrayfire-users
* ArrayFire Services:  [Consulting](http://arrayfire.com/consulting/)  |  [Support](http://arrayfire.com/support/)   |  [Training](http://arrayfire.com/training/)
* ArrayFire Blogs: http://arrayfire.com/blog/
* Email: <mailto:technical@arrayfire.com>
