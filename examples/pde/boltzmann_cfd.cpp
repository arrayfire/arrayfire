/*******************************************************
 * Copyright (c) 2023, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

/*
    This is a Computational Fluid Dynamics Simulation using the Lattice
   Boltzmann Method For this simulation we are using D2N9 (2 dimensions, 9
   neighbors) with bounce-back boundary conditions For more information on the
   simulation equations, check out
   https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods#Mathematical_equations_for_simulations

    The initial conditions of the fluid are obtained from three images that
   specify their properties using the function read_initial_condition_arrays.
   These images can be modified to simulate different cases
*/

#include <arrayfire.h>
#include <chrono>
#include <iostream>
#include <thread>

/*
    Values of the D2N9 grid follow the following order structure:


          -1      0       1
      * ----------------------> x
  -1   |   6      3       0
       |
   0   |   7      4       1
       |
   1   |   8      5       2
       |
       v
       y

    The (-1, 0, 1) refer to the x and y offsets with respect to a single cell
    and the (0-8) refer to indices of each cell in the 3x3 grid

    Eg. Element with index 4 is the center of the grid which has an x-offset =
  ex_vals[4] = 0 and y-offset = ey_vals[4] = 0 with its quantities being
  weighted with weight wt_vals[4] = 16/36
*/

static const float ex_vals[] = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0};

static const float ey_vals[] = {1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0};

static const float wt_vals[] = {1.0f / 36.0f, 4.0f / 36.0f,  1.0f / 36.0f,
                                4.0f / 36.0f, 16.0f / 36.0f, 4.0f / 36.0f,
                                1.0f / 36.0f, 4.0f / 36.0f,  1.0f / 36.0f};

static const int opposite_indices[] = {8, 7, 6, 5, 4, 3, 2, 1, 0};

struct Simulation {
    // Fluid quantities
    af::array ux;
    af::array uy;
    af::array rho;
    af::array sigma;
    af::array f;
    af::array feq;

    // Constant velocity boundary conditions positions
    af::array set_boundaries;

    // Simulation Parameters
    size_t grid_width;
    size_t grid_height;
    float density;
    float velocity;
    float reynolds;

    // Helper arrays stored for computation
    af::array ex;
    af::array ey;
    af::array wt;

    af::array ex_T;
    af::array ey_T;
    af::array wt_T;

    af::array ex_;
    af::array ey_;
};

/**
 * @brief Create a simulation object containing all the initial parameters and
 * condition of the simulation
 *
 * @details
 * For the ux, uy, and boundary images, we use RGB values for to define the
 * specific quantites for each grid cell/pixel
 *
 * /// R & B for ux & uy
 *
 * For ux and uy, Red means positive value while Blue means negative value. The
 * speed value for both ux and uy is computed as $(R - B) * velocity / 255$.
 *
 * For example, for the same pixel in the two images if we had ux = RGB(255,0,0)
 * and uy = RGB(0,0,255) means that cell's fluid has an x-velocity of +v and
 * y-velocity of -v where v is the velocity quantity pass to this function.
 *
 * Note that having the same value in the R and B components will cancel each
 * other out, i.e., have the fluid has 0 velocity in that direction similar to
 * having it be 0.
 *
 * /// G for ux & uy
 *
 * The G component is reserved for an object or obstacle. Any non-zero value for
 * the green component represents a hard boundary in the simulation
 *
 * /// RGB for boundary
 *
 * Any non-zero value for any of the components in the RGB value of the pixel
 * means that the initial values passed for ux and uy will remain constant
 * throught the simulation
 *
 */
Simulation create_simulation(uint32_t grid_width, uint32_t grid_height,
                             float density, float velocity, float reynolds,
                             const char* ux_image_filename,
                             const char* uy_image_filename,
                             const char* boundaries_filename) {
    Simulation sim;

    sim.grid_width  = grid_width;
    sim.grid_height = grid_height;
    sim.velocity    = velocity;
    sim.density     = density;
    sim.reynolds    = reynolds;

    try {
        sim.ux = af::loadImage(ux_image_filename, true);
    } catch (const af::exception& e) {
        std::cerr << e.what() << std::endl;
        sim.ux = af::constant(0, grid_width, grid_height, 3);
    }

    auto ux_dim = sim.ux.dims();
    if (ux_dim[0] != grid_width || ux_dim[1] != grid_height) {
        std::cerr
            << "Fluid flow ux image has dimensions different to the simulation"
            << std::endl;
        throw std::runtime_error{
            "Fluid flow ux image has dimensions different to the simulation"};
    }

    try {
        sim.uy = af::loadImage(uy_image_filename, true);
    } catch (const af::exception& e) {
        std::cerr << e.what() << std::endl;
        sim.uy = af::constant(0, grid_width, grid_height, 3);
    }

    auto uy_dim = sim.uy.dims();
    if (uy_dim[0] != grid_width || uy_dim[1] != grid_height) {
        std::cerr
            << "Fluid flow uy image has dimensions different to the simulation"
            << std::endl;
        throw std::runtime_error{
            "Fluid flow uy image has dimensions different to the simulation"};
    }

    try {
        sim.set_boundaries = af::loadImage(boundaries_filename, false);
    } catch (const af::exception& e) {
        std::cerr << e.what() << std::endl;
        sim.set_boundaries = af::constant(0, grid_width, grid_height);
    }

    auto b_dim = sim.set_boundaries.dims();
    if (b_dim[0] != grid_width || b_dim[1] != grid_height) {
        std::cerr
            << "Fluid boundary image has dimensions different to the simulation"
            << std::endl;
        throw std::runtime_error{
            "Fluid boundary image has dimensions different to the simulation"};
    }

    sim.ux = (sim.ux(af::span, af::span, 0).T() -
              sim.ux(af::span, af::span, 2).T()) *
             velocity / 255.f;
    sim.uy = (sim.uy(af::span, af::span, 0).T() -
              sim.uy(af::span, af::span, 2).T()) *
             velocity / 255.f;
    sim.set_boundaries = sim.set_boundaries.T() > 0;

    return sim;
}

/**
 * @brief Initializes internal values used for computation
 *
 */
void initialize(Simulation& sim) {
    auto& ux    = sim.ux;
    auto& uy    = sim.uy;
    auto& rho   = sim.rho;
    auto& sigma = sim.sigma;
    auto& f     = sim.f;
    auto& feq   = sim.feq;

    auto& ex   = sim.ex;
    auto& ey   = sim.ey;
    auto& wt   = sim.wt;
    auto& ex_  = sim.ex_;
    auto& ey_  = sim.ey_;
    auto& ex_T = sim.ex_T;
    auto& ey_T = sim.ey_T;
    auto& wt_T = sim.wt_T;

    auto density  = sim.density;
    auto velocity = sim.velocity;
    auto xcount   = sim.grid_width;
    auto ycount   = sim.grid_height;

    ex = af::array(1, 1, 9, ex_vals);
    ey = af::array(1, 1, 9, ey_vals);
    wt = af::array(1, 1, 9, wt_vals);

    ex_T = af::array(1, 9, ex_vals);
    ey_T = af::array(1, 9, ey_vals);
    wt_T = af::moddims(wt, af::dim4(1, 9));

    rho   = af::constant(density, xcount, ycount, f32);
    sigma = af::constant(0, xcount, ycount, f32);

    f = af::constant(0, xcount, ycount, 9, f32);

    ex_ = af::tile(ex, xcount, ycount, 1);
    ey_ = af::tile(ey, xcount, ycount, 1);

    // Initialization of the distribution function
    auto edotu = ex_ * ux + ey_ * uy;
    auto udotu = ux * ux + uy * uy;

    feq = rho * wt *
          ((edotu * edotu * 4.5f) - (udotu * 1.5f) + (edotu * 3.0f) + 1.0f);
    f = feq;
}

/**
 * @brief Updates the particle distribution functions for the new simulation
 * frame
 *
 */
void collide_stream(Simulation& sim) {
    auto& ux             = sim.ux;
    auto& uy             = sim.uy;
    auto& rho            = sim.rho;
    auto& sigma          = sim.sigma;
    auto& f              = sim.f;
    auto& feq            = sim.feq;
    auto& set_boundaries = sim.set_boundaries;

    auto& ex   = sim.ex;
    auto& ey   = sim.ey;
    auto& wt   = sim.wt;
    auto& ex_  = sim.ex_;
    auto& ey_  = sim.ey_;
    auto& ex_T = sim.ex_T;
    auto& ey_T = sim.ey_T;
    auto& wt_T = sim.wt_T;

    auto density  = sim.density;
    auto velocity = sim.velocity;
    auto reynolds = sim.reynolds;
    auto xcount   = sim.grid_width;
    auto ycount   = sim.grid_height;

    const float viscosity =
        velocity * std::sqrt(static_cast<float>(xcount * ycount)) / reynolds;
    const float tau  = 0.5f + 3.0f * viscosity;
    const float csky = 0.16f;

    auto edotu = ex_ * ux + ey_ * uy;
    auto udotu = ux * ux + uy * uy;

    // Compute the new distribution function
    feq =
        rho * wt * (edotu * edotu * 4.5f - udotu * 1.5f + edotu * 3.0f + 1.0f);

    auto taut =
        af::sqrt(sigma * (csky * csky * 18.0f * 0.25f) + (tau * tau * 0.25f)) -
        (tau * 0.5f);

    // Compute the shifted distribution functions
    auto fplus = f - (f - feq) / (taut + tau);

    // Compute new particle distribution according to the corresponding D2N9
    // weights
    for (int i = 0; i < 9; ++i) {
        int xshift = static_cast<int>(ex_vals[i]);
        int yshift = static_cast<int>(ey_vals[i]);

        fplus(af::span, af::span, i) =
            af::shift(fplus(af::span, af::span, i), xshift, yshift);
    }

    // Keep the boundary conditions at the borders the same
    af::replace(fplus, af::tile(!set_boundaries, af::dim4(1, 1, 9)), f);

    // Update the particle distribution
    f = fplus;

    // Computing u dot e at the each of the boundaries
    af::array ux_top = ux.rows(0, 2);
    ux_top =
        af::moddims(af::tile(ux_top, af::dim4(1, 3)).T(), af::dim4(ycount, 9));
    af::array ux_bot = ux.rows(xcount - 3, xcount - 1);
    ux_bot =
        af::moddims(af::tile(ux_bot, af::dim4(1, 3)).T(), af::dim4(ycount, 9));

    af::array uy_top = uy.rows(0, 2);
    uy_top =
        af::moddims(af::tile(uy_top, af::dim4(1, 3)).T(), af::dim4(ycount, 9));
    af::array uy_bot = uy.rows(xcount - 3, xcount - 1);
    uy_bot =
        af::moddims(af::tile(uy_bot, af::dim4(1, 3)).T(), af::dim4(ycount, 9));

    auto ux_lft = af::tile(ux.cols(0, 2), af::dim4(1, 3));
    auto uy_lft = af::tile(uy.cols(0, 2), af::dim4(1, 3));
    auto ux_rht = af::tile(ux.cols(ycount - 3, ycount - 1), af::dim4(1, 3));
    auto uy_rht = af::tile(uy.cols(ycount - 3, ycount - 1), af::dim4(1, 3));

    auto ubdoute_top = ux_top * ex_T + uy_top * ey_T;
    auto ubdoute_bot = ux_bot * ex_T + uy_bot * ey_T;
    auto ubdoute_lft = ux_lft * ex_T + uy_lft * ey_T;
    auto ubdoute_rht = ux_rht * ex_T + uy_rht * ey_T;

    // Computing bounce-back boundary conditions
    auto fnew_top = af::moddims(fplus.row(1), af::dim4(ycount, 9)) -
                    6.0 * density * wt_T * ubdoute_top;
    auto fnew_bot = af::moddims(fplus.row(xcount - 2), af::dim4(ycount, 9)) -
                    6.0 * density * wt_T * ubdoute_bot;
    auto fnew_lft = af::moddims(fplus.col(1), af::dim4(xcount, 9)) -
                    6.0 * density * wt_T * ubdoute_lft;
    auto fnew_rht = af::moddims(fplus.col(ycount - 2), af::dim4(xcount, 9)) -
                    6.0 * density * wt_T * ubdoute_rht;

    // Update the values near the boundaries with the correct bounce-back
    // boundary
    for (int i = 0; i < 9; ++i) {
        int xshift = static_cast<int>(ex_vals[i]);
        int yshift = static_cast<int>(ey_vals[i]);
        if (xshift == 1)
            f(1, af::span, opposite_indices[i]) = fnew_top(af::span, i);
        if (xshift == -1)
            f(xcount - 2, af::span, opposite_indices[i]) =
                fnew_bot(af::span, i);
        if (yshift == 1)
            f(af::span, 1, opposite_indices[i]) = fnew_lft(af::span, i);
        if (yshift == -1)
            f(af::span, ycount - 2, opposite_indices[i]) =
                fnew_rht(af::span, i);
    }
}

/**
 * @brief Updates the velocity field, density and strain at each point in the
 * grid
 *
 */
void update(Simulation& sim) {
    auto& ux    = sim.ux;
    auto& uy    = sim.uy;
    auto& rho   = sim.rho;
    auto& sigma = sim.sigma;
    auto& f     = sim.f;
    auto& feq   = sim.feq;
    auto& ex    = sim.ex;
    auto& ey    = sim.ey;

    auto e_tile = af::join(3, af::constant(1, 1, 1, 9), ex, ey);
    auto result = af::sum(f * e_tile, 2);

    rho = result(af::span, af::span, af::span, 0);
    result /= rho;
    ux = result(af::span, af::span, af::span, 1);
    uy = result(af::span, af::span, af::span, 2);

    // Above code equivalent to
    // rho = af::sum(f, 2);
    // ux = af::sum(f * ex, 2) / rho;
    // uy = af::sum(f * ey, 2) / rho;

    auto product   = f - feq;
    auto e_product = af::join(3, ex * ex, ex * ey * std::sqrt(2), ey * ey);

    sigma = af::sqrt(af::sum(af::pow(af::sum(product * e_product, 2), 2), 3));

    // Above code equivalent to

    // auto xx = af::sum(product * ex * ex, 2);
    // auto xy = af::sum(product * ex * ey, 2);
    // auto yy = af::sum(product * ey * ey, 2);

    // sigma = af::sqrt(xx * xx + xy * xy * 2 + yy * yy);
}

af::array generate_image(size_t width, size_t height, const Simulation& sim) {
    const auto& ux         = sim.ux;
    const auto& uy         = sim.uy;
    const auto& boundaries = sim.set_boundaries;
    auto velocity          = sim.velocity;

    float image_scale =
        static_cast<float>(width) / static_cast<float>(sim.grid_width - 1);

    // Relative Flow speed at each cell
    auto val = af::sqrt(ux * ux + uy * uy) / velocity;

    af::replace(val, val != 0 || !boundaries, -1.0);

    // Scaling and interpolating flow speed to the window size
    if (width != sim.grid_width || height != sim.grid_height)
        val =
            af::approx2(val, af::iota(width, af::dim4(1, height)) / image_scale,
                        af::iota(height, af::dim4(1, width)).T() / image_scale);

    // Flip image
    val = val.T();

    auto image  = af::constant(0, height, width, 3);
    auto image2 = image;

    // Add custom coloring
    image(af::span, af::span, 0) = val * 2;
    image(af::span, af::span, 1) = val * 2;
    image(af::span, af::span, 2) = 1.0 - val * 2;

    image2(af::span, af::span, 0) = 1;
    image2(af::span, af::span, 1) = -2 * val + 2;
    image2(af::span, af::span, 2) = 0;

    auto tile_val = af::tile(val, 1, 1, 3);
    af::replace(image, tile_val < 0.5, image2);
    af::replace(image, tile_val >= 0, 0.0);

    return image;
}

void lattice_boltzmann_cfd_demo() {
    // Define the lattice for the simulation
    const size_t len         = 128;
    const size_t grid_width  = len;
    const size_t grid_height = len;

    // Specify the image scaling displayed
    float scale = 4.0f;

    // Forge window initialization
    int height = static_cast<int>(grid_width * scale);
    int width  = static_cast<int>(grid_height * scale);
    af::Window window(height, width, "Driven Cavity Flow");

    int frame_count       = 0;
    int max_frames        = 20000;
    int simulation_frames = 100;
    float total_time      = 0;
    float total_time2     = 0;

    // CFD fluid parameters
    const float density  = 2.7f;
    const float velocity = 0.35f;
    const float reynolds = 1e5f;

    const char* ux_image = ASSETS_DIR "/examples/images/default_ux.bmp";
    const char* uy_image = ASSETS_DIR "/examples/images/default_uy.bmp";
    const char* set_boundary_image =
        ASSETS_DIR "/examples/images/default_boundary.bmp";

    // Tesla Valve Fluid Simulation - entering from constricted side
    {
        //           ux_image = ASSETS_DIR "/examples/images/left_tesla_ux.bmp";
        //           uy_image = ASSETS_DIR "/examples/images/left_tesla_uy.bmp";
        // set_boundary_image = ASSETS_DIR
        // "/examples/images/left_tesla_boundary.bmp";
    }

    // Tesla Valve Fluid Simulation - entering from transfer side
    {
        //           ux_image = ASSETS_DIR
        //           "/examples/images/right_tesla_ux.bmp"; uy_image =
        //           ASSETS_DIR "/examples/images/right_tesla_uy.bmp";
        // set_boundary_image = ASSETS_DIR
        // "/examples/images/right_tesla_boundary.bmp";
    }

    // Reads the initial values of fluid quantites and simulation parameters
    Simulation sim =
        create_simulation(grid_width, grid_height, density, velocity, reynolds,
                          ux_image, uy_image, set_boundary_image);

    // Initializes the simulation quantites
    initialize(sim);

    while (!window.close() && frame_count != max_frames) {
        af::sync();
        auto begin = std::chrono::high_resolution_clock::now();

        // Computes the new particle distribution functions for the new
        // simulation frame
        collide_stream(sim);

        // Updates the velocity, density, and stress fields
        update(sim);

        af::sync();
        auto end = std::chrono::high_resolution_clock::now();

        // Calculate computation time of 1 simulation frame
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
                .count();

        // Used for computing the distribution of frame computation time
        total_time += duration;
        total_time2 += duration * duration;

        // Every number of `simulation_frames` display the last computed frame
        // to the screen
        if (frame_count % simulation_frames == 0) {
            auto image = generate_image(width, height, sim);

            // Display colored image
            window.image(image);

            float avg_time  = total_time / (float)simulation_frames;
            float stdv_time = std::sqrt(total_time2 * simulation_frames -
                                        total_time * total_time) /
                              (float)simulation_frames;

            std::cout << "Average Simulation Step Time: (" << avg_time
                      << " +/- " << stdv_time
                      << ") us; Total simulation time: " << total_time
                      << " us; Simulation Frames: " << simulation_frames
                      << std::endl;

            total_time  = 0;
            total_time2 = 0;
        }

        frame_count++;
    }
}

int main(int argc, char** argv) {
    int device = argc > 1 ? std::atoi(argv[1]) : 0;

    try {
        af::setDevice(device);
        af::info();

        std::cout << "** ArrayFire CFD Simulation Demo\n\n";

        lattice_boltzmann_cfd_demo();
    } catch (const af::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}