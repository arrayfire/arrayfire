// Generate random bits and send them to STDOUT.
// Suitable for testing with PractRand, c.f. http://pracrand.sourceforge.net/
// and http://www.pcg-random.org/posts/how-to-test-with-practrand.html
// Commandline arguments: backend, device, rng_type
// Example:
// random_practrand 0 0 200 | RNG_test stdin32
#include <arrayfire.h>
#include <cstdint>
#include <cstdio>

int main(int argc, char **argv) {
    int backend = argc > 1 ? atoi(argv[1]) : 0;
    setBackend(static_cast<Backend>(backend));
    int device = argc > 2 ? atoi(argv[2]) : 0;
    setDevice(device);
    int rng = argc > 3 ? atoi(argv[3]) : 100;
    setDefaultRandomEngineType(static_cast<randomEngineType>(rng));

    setSeed(0xfe47fe0cc078ec30ULL);
    int samples = 1024 * 1024;
    while (1) {
        array values      = randu(samples, u32);
        uint32_t *pvalues = values.host<uint32_t>();
        fwrite((void *)pvalues, samples * sizeof(*pvalues), 1, stdout);
        freeHost(pvalues);
    }
}
