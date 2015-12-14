#include <math.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <arrayfire.h>
#include "../common/progress.h"

using namespace af;

Window *win;

array normalize(array a, float max)
{
    float mx = max * 0.5;
    float mn = -max * 0.5;
    return (a-mn)/(mx-mn);
}

static void swe(bool console)
{
    double time_total = 20; // run for N seconds
    // Grid length, number and spacing
    const unsigned Lx = 512, nx = Lx + 1;
    const unsigned Ly = 512, ny = Ly + 1;
    const float dx = Lx / (nx - 1);
    const float dy = Ly / (ny - 1);

    array ZERO = constant(0, nx, ny);
    array um = ZERO, vm = ZERO;
    unsigned io = (unsigned)floor(Lx  / 5.0f),
             jo = (unsigned)floor(Ly / 5.0f),
             k = 20;
    array x = tile(moddims(seq(nx),nx,1), 1,ny);
    array y = tile(moddims(seq(ny),1,ny), nx,1);

    // Initial condition
    array etam = 0.01f * exp((-((x - io) * (x - io) + (y - jo) * (y - jo))) / (k * k));
    float m_eta = max<float>(etam);
    array eta = etam;
    float dt = 0.5;

    // conv kernels
    float h_diff_kernel[] = {9.81f * (dt / dx), 0, -9.81f * (dt / dx)};
    float h_lap_kernel[] = {0, 1, 0, 1, -4, 1, 0, 1, 0};

    array h_diff_kernel_arr(3, h_diff_kernel);
    array h_lap_kernel_arr(3, 3, h_lap_kernel);

    if(!console) {
        win = new Window(512, 512,"Shallow Water Equations");
        win->setColorMap(AF_COLORMAP_MOOD);
    }

    timer t = timer::start();
    unsigned iter = 0;
    while (progress(iter, t, time_total)) {
        // compute
        array up = um + convolve(eta, h_diff_kernel_arr);
        array vp = um + convolve(eta, h_diff_kernel_arr.T());
        array e = convolve(eta, h_lap_kernel_arr);
        array etap = 2 * eta - etam + (2 * dt * dt) / (dx * dy) * e;

        etam = eta;
        eta = etap;
        if (!console) {
            win->image(normalize(eta, m_eta));
            // viz
        } else eval(eta, up, vp);
        iter++;
    }
}
int main(int argc, char* argv[])
{
    int device = argc > 1 ? atoi(argv[1]) : 0;
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
