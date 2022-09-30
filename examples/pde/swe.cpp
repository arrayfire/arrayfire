#include <arrayfire.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace af;

Window* win;

array normalize(array a, float max) {
    float mx = max * 0.5;
    float mn = -max * 0.5;
    return (a - mn) / (mx - mn);
}

static void swe(bool console) {
    // Grid length, number and spacing
    const unsigned Lx = 1600, nx = Lx + 1;
    const unsigned Ly = 1600, ny = Ly + 1;
    const float dx = Lx / (nx - 1);
    const float dy = Ly / (ny - 1);

    array ZERO = constant(0, nx, ny);
    array um = ZERO, vm = ZERO;
    unsigned io = (unsigned)floor(Lx / 6.0f), jo = (unsigned)floor(Ly / 6.0f),
             k = 15;
    array x    = tile(range(nx), 1, ny);
    array y    = tile(range(dim4(1, ny), 1), nx, 1);

    // initial condition
    array etam =
        0.01f * exp((-((x - io) * (x - io) + (y - jo) * (y - jo))) / (k * k));
    float m_eta = max<float>(etam);
    array eta   = etam;
    float dt    = 0.5;

    // conv kernels
    float h_diff_kernel[] = {9.81f * (dt / dx), 0, -9.81f * (dt / dx)};
    float h_lap_kernel[]  = {0, 1, 0, 1, -4, 1, 0, 1, 0};

    array h_diff_kernel_arr(3, h_diff_kernel);
    array h_lap_kernel_arr(3, 3, h_lap_kernel);

    if (!console) {
        win = new Window(1536, 768, "Shallow Water Equations");
        win->grid(2, 2);
    }

    unsigned iter            = 0;
    unsigned random_interval = 30;

    while (!win->close()) {
        if (iter > 2000) {
            // Initial condition
            etam  = 0.01f * exp((-((x - io) * (x - io) + (y - jo) * (y - jo))) /
                                (k * k));
            m_eta = max<float>(etam);
            eta   = etam;
            iter  = 0;
        }

        // raindrops
        if (iter % 100 == 0 || iter % 130 == 0 || iter % random_interval == 0) {
            unsigned io     = (unsigned)floor(rand() % Lx),
                     jo     = (unsigned)floor(rand() % Ly);
            random_interval = rand() % 200 + 1;
            eta += 0.01f * exp((-((x - io) * (x - io) + (y - jo) * (y - jo))) /
                               (k * k));
        }

        // compute
        array up   = um + convolve(eta, h_diff_kernel_arr);
        array vp   = um + convolve(eta, h_diff_kernel_arr.T());
        array e    = convolve(eta, h_lap_kernel_arr);
        array etap = 2 * eta - etam + (2 * dt * dt) / (dx * dy) * e;

        etam = eta;
        eta  = etap;

        m_eta = max<float>(etam);
        if (!console) {
            (*win)(0, 0).setColorMap(AF_COLORMAP_BLUE);
            array hist_out = histogram(normalize(eta, m_eta), 15);

            (*win)(0, 1).setAxesLimits(0, hist_out.elements(), 0,
                                       max<float>(hist_out));

            (*win)(0, 0).image(normalize(eta, m_eta));
            (*win)(0, 1).hist(hist_out, 0, 1,
                              "Normalized Pressure Distribution");
            (*win)(1, 0).plot(range(up.dims(1)), vp.col(0),
                              "Pressure at left boundary");
            (*win)(1, 1).plot(
                flat(eta.col(0)), flat(up.col(0)), flat(vp.col(0)),
                "Gradients versus Magnitude at left boundary");  // viz
            win->show();
        } else
            eval(eta, up, vp);
        iter++;
    }
}

int main(int argc, char* argv[]) {
    int device   = argc > 1 ? atoi(argv[1]) : 0;
    bool console = argc > 2 ? argv[2][0] == '-' : false;
    try {
        af::setDevice(device);
        af::info();
        printf("Simulation of shallow water equations\n");
        swe(console);
    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
