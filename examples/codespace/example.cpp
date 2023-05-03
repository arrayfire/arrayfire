#include <arrayfire.h>             // > > > > > > > > > > > CLICK PLAY UPPER RIGHT
const int N = (1 << 10);
af::array A; void fn() { auto B = af::fft2(A); af::eval(B); }

int main(int argc, char** argv) {
    af::info();                                          // Show compute device info
    A = af::randu(N, N);                                 // Generates uniform random matrix
    af::array B = af::fft2(A);                           // Compute 2D FFT
    af::array C = B(af::seq(0, 5), af::seq(0, 5));       // Grab 5x5 top left corner
    af::array D = af::pinverse(C);                       // Pseudoinverse is easy
    af_print(af::real(D));                               // Nicely see the results
    printf("2D FFT %dx%d: %4.0f Gflops\n", N, N,         // Do a benchmark
             10.0 * N * N * 10 / (af::timeit(fn) * 1e9));
}