/*******************************************************
 * Copyright (c) 2024, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

/*
    This is a Black Hole Raytracer.
    For this raytracer we are using backwards path tracing to compute the
   resulting image The path of the rays shot from the camera are simulated step
   by step from the null geodesics light follows in spacetime. The geodesics are
   computed from the spacetime metric of the space. This project has three
   metrics that can be used: Schwarzchild, Kerr, and Ellis.

    For more information on the black hole raytracing, check out
    Riazuelo, A. (2015). Seeing relativity -- I. Ray tracing in a Schwarzschild
   metric to explore the maximal analytic extension of the metric and making a
   proper rendering of the stars. ArXiv.
   https://doi.org/10.1142/S0218271819500421

    For more information on raytracing, check out
    Raytracing in a Weekend Series, https://raytracing.github.io/

    Image being used for the background is Westerlund 2 from
    NASA, ESA, the Hubble Heritage Team (STScI/AURA), A. Nota (ESA/STScI), and
   the Westerlund 2 Science Team See
   http://www.spacetelescope.org/images/heic1509a/ for details.

    The default scene is the rotating black hole using the Kerr metric set by
   the global variable 'scene' The parameters of the blackholes/wormholes may be
   changed at the top with the simulation constants The parameters of the image
   may be changed in the 'raytracing' function.
*/
#include <arrayfire.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

enum class Scene { ROTATE_BH, STATIC_BH, WORMHOLE };

// Scene being computed
static constexpr Scene scene = Scene::ROTATE_BH;

// **** Simulation Constants ****
static constexpr double M = 0.5;   // Black Hole Mass
static constexpr double J = 0.24;  // Black Hole Rotation (J < M^2)
static constexpr double b = 3.0;   // Wormhole drainhole parameter

/**
 * @brief Generates a string progress bar
 *
 * @param current current job
 * @param total total number of jobs
 * @param start_info progress bar prior info
 */
void status_bar(int current, int total, const std::string& start_info) {
    auto precision        = std::cout.precision();
    static auto prev_time = std::chrono::high_resolution_clock::now();
    static auto prev      = current - 1;

    auto curr_time = std::chrono::high_resolution_clock::now();

    double percent  = 100.0 * (double)(current + 1) / (double)total;
    std::string str = "[";
    for (int i = 0; i < 50; ++i) {
        if (percent >= i * 2)
            str += "=";
        else
            str += " ";
    }
    str += "]";

    auto time = (total - current) * (curr_time - prev_time) / (current - prev);

    if (current != total) {
        using namespace std::chrono_literals;
        std::cout << start_info << " " << std::fixed << std::setprecision(1)
                  << percent << "%  " << str << " Time Remaining: ";
        if (std::chrono::duration_cast<std::chrono::seconds>(time).count() >
            300)
            std::cout << std::chrono::duration_cast<std::chrono::minutes>(time)
                             .count()
                      << " min";
        else
            std::cout << std::chrono::duration_cast<std::chrono::seconds>(time)
                             .count()
                      << " s";

        std::cout << std::string(5, ' ') << '\r';
    } else
        std::cout << "\rDone!" << std::string(120, ' ') << std::endl;

    std::cout << std::setprecision(precision) << std::defaultfloat;
}

/**
 * @brief Returns the euclidean dot product for two cartesian vectors with 3
 * coords
 *
 * @param lhs
 * @param rhs
 * @return af::array
 */
af::array dot3(const af::array& lhs, const af::array& rhs) {
    return af::sum(lhs * rhs, 0);
}

/**
 * @brief Returns the euclidean norm for a cartesian vector with 3 coords
 *
 * @param vector
 * @return af::array
 */
af::array norm3(const af::array& vector) {
    return af::sqrt(dot3(vector, vector));
}

/**
 * @brief Returns the normalized vector for a cartesian vector with 3 coords
 *
 * @param vector
 * @return af::array
 */
af::array normalize3(const af::array& vector) { return vector / norm3(vector); }

af::exception make_error(const char* string) {
    std::cout << string << std::endl;
    return af::exception(string);
}

/**
 * @brief Transforms degrees to radians
 *
 * @param degrees
 * @return double
 */
double radians(double degrees) { return degrees * af::Pi / 180.0; }

/**
 * @brief Computes the cross_product of two euclidean vectors
 *
 * @param lhs
 * @param rhs
 * @return af::array
 */
af::array cross_product(const af::array& lhs, const af::array& rhs) {
    if (lhs.dims() != rhs.dims())
        throw make_error("Arrays must have the same dimensions");
    else if (lhs.dims()[0] != 3)
        throw make_error("Arrays must have 3 principal coordintes");

    return af::join(
        0,
        lhs(1, af::span, af::span) * rhs(2, af::span, af::span) -
            lhs(2, af::span, af::span) * rhs(1, af::span, af::span),
        lhs(2, af::span, af::span) * rhs(0, af::span, af::span) -
            lhs(0, af::span, af::span) * rhs(2, af::span, af::span),
        lhs(0, af::span, af::span) * rhs(1, af::span, af::span) -
            lhs(1, af::span, af::span) * rhs(0, af::span, af::span));
}

/**
 * @brief Transform the position vectors from cartesian to spherical coordinates
 *
 * @param pos
 * @return af::array
 */
af::array cart_to_sph_position(const af::array& pos) {
    if (pos.dims()[0] != 3)
        throw make_error("Arrays must have 3 principal coordintes");

    af::array x = pos(0, af::span);
    af::array y = pos(1, af::span);
    af::array z = pos(2, af::span);

    af::array r = af::sqrt(x * x + y * y + z * z);
    af::array o = af::acos(z / r);
    af::array p = af::atan2(y, x);

    af::array transformed_pos = af::join(0, r, o, p);

    return transformed_pos;
}

/**
 * @brief Transform the velocity vectors from cartesian to spherical coordinates
 *
 * @param vel
 * @param pos
 * @return af::array
 */
af::array cart_to_sph_velocity(const af::array& vel, const af::array& pos) {
    if (vel.dims() != pos.dims())
        throw make_error("Arrays must have the same dimensions");
    else if (pos.dims()[0] != 3)
        throw make_error("Arrays must have 3 principal coordintes");

    af::array x = pos(0, af::span);
    af::array y = pos(1, af::span);
    af::array z = pos(2, af::span);

    af::array r = af::sqrt(x * x + y * y + z * z);
    af::array o = af::acos(z / r);
    af::array p = af::atan2(y, x);

    af::array ux = vel(0, af::span);
    af::array uy = vel(1, af::span);
    af::array uz = vel(2, af::span);

    af::array ur = (ux * x + uy * y + uz * z) / r;
    af::array up = (uy * af::cos(p) - ux * af::sin(p)) / r;
    af::array uo =
        (af::cos(o) * (ux * af::cos(p) + uy * af::sin(p)) - uz * af::sin(o)) /
        r;
    af::array transformed_vel = af::join(0, ur, uo, up);

    return transformed_vel;
}

/**
 * @brief Transform the position vectors from spherical to cartesian coordinates
 *
 * @param pos
 * @return af::array
 */
af::array sph_to_cart_position3(const af::array& pos) {
    af::array r = pos(0, af::span);
    af::array o = pos(1, af::span);
    af::array p = pos(2, af::span);

    af::array x = r * af::sin(o) * af::cos(p);
    af::array y = r * af::sin(o) * af::sin(p);
    af::array z = r * af::cos(o);

    af::array transformed_pos = af::join(0, x, y, z);

    return transformed_pos;
}

/**
 * @brief Computes the inverse of a 4x4 matrix with the layout
 *          [ a 0 0 b ]
 *          [ 0 c 0 0 ]
 *          [ 0 0 d 0 ]
 *          [ b 0 0 e ]
 *
 * @param metric af::array with the shape af::dims4(4, 4, M, N)
 *
 * @return af::array with the shape af::dims4(4, 4, M, N)
 */
af::array inv_metric(const af::array& metric) {
    af::array a = metric(0, 0, af::span);
    af::array b = metric(3, 0, af::span);
    af::array c = metric(1, 1, af::span);
    af::array d = metric(2, 2, af::span);
    af::array e = metric(3, 3, af::span);

    af::array det = b * b - a * e;

    auto res = af::constant(0, 4, 4, metric.dims()[2], metric.dims()[3], f64);

    res(0, 0, af::span) = -e / det;
    res(0, 3, af::span) = b / det;
    res(3, 0, af::span) = b / det;
    res(1, 1, af::span) = 1.0 / c;
    res(2, 2, af::span) = 1.0 / d;
    res(3, 3, af::span) = -a / det;

    return res;
}

/**
 * @brief Computes the 4x4 metric matrix for the given 4-vector positions
 *
 * @param pos af::dim4(4, N)
 * @return af::array af::dim4(4, 4, 1, N)
 */
af::array metric4(const af::array& pos) {
    if (pos.dims()[0] != 4)
        throw make_error("Arrays must have 4 principal coordinates");

    auto dims = pos.dims();

    af::array t = af::moddims(pos(0, af::span), 1, 1, dims[1]);
    af::array r = af::moddims(pos(1, af::span), 1, 1, dims[1]);
    af::array o = af::moddims(pos(2, af::span), 1, 1, dims[1]);
    af::array p = af::moddims(pos(3, af::span), 1, 1, dims[1]);

    af::array gtt, gtr, gto, gtp, grt, grr, gro, grp, got, gor, goo, gop, gpt,
        gpr, gpo, gpp;

    switch (scene) {
        // ******* Kerr Black Hole Metric *******
        case Scene::ROTATE_BH: {
            auto rs    = 2.0 * M;
            auto a     = J / M;
            auto delta = (r - rs) * r + a * a;
            auto sigma = r * r + af::pow(a * af::cos(o), 2);

            gtt = 1.0 - r * rs / sigma;
            gtr = af::constant(0.0, 1, 1, dims[1], f64);
            gto = af::constant(0.0, 1, 1, dims[1], f64);
            gtp = rs * r * a * af::pow(af::sin(o), 2.0) / sigma;
            grr = -sigma / delta;
            gro = af::constant(0.0, 1, 1, dims[1], f64);
            grp = af::constant(0.0, 1, 1, dims[1], f64);
            goo = -sigma;
            gop = af::constant(0.0, 1, 1, dims[1], f64);
            gpp =
                -(r * r + a * a + rs * r * af::pow(a * af::sin(o), 2) / sigma) *
                af::pow(af::sin(o), 2);

            break;
        }

        // ******* Schwarzchild Black Hole Metric *******
        case Scene::STATIC_BH: {
            gtt = 1.0 - 2.0 * M / r;
            gtr = af::constant(0.0, 1, 1, dims[1], f64);
            gto = af::constant(0.0, 1, 1, dims[1], f64);
            gtp = af::constant(0.0, 1, 1, dims[1], f64);
            grr = -1.0 / (1.0 - 2.0 * M / r);
            gro = af::constant(0.0, 1, 1, dims[1], f64);
            grp = af::constant(0.0, 1, 1, dims[1], f64);
            goo = -r * r;
            gop = af::constant(0.0, 1, 1, dims[1], f64);
            gpp = -af::pow(r * af::sin(o), 2);

            break;
        }

        // ******* Ellis Wormhole Metric *******
        case Scene::WORMHOLE: {
            gtt = af::constant(1.0, 1, 1, dims[1], f64);
            gtr = af::constant(0.0, 1, 1, dims[1], f64);
            gto = af::constant(0.0, 1, 1, dims[1], f64);
            gtp = af::constant(0.0, 1, 1, dims[1], f64);
            grr = -af::constant(1.0, 1, 1, dims[1], f64);
            gro = af::constant(0.0, 1, 1, dims[1], f64);
            grp = af::constant(0.0, 1, 1, dims[1], f64);
            goo = -(r * r + b * b);
            gop = af::constant(0.0, 1, 1, dims[1], f64);
            gpp = -(r * r + b * b) * af::pow(af::sin(o), 2);

            break;
        }

        default: throw;
    }

    auto res = af::join(
        0, af::join(1, gtt, gtr, gto, gtp), af::join(1, gtr, grr, gro, grp),
        af::join(1, gto, gro, goo, gop), af::join(1, gtp, grp, gop, gpp));

    return res;
}

/**
 * @brief Computes the dot product as defined by a metric between two 4-vector
 * velocities
 *
 * @param pos
 * @param lhs
 * @param rhs
 * @return af::array
 */
af::array dot_product(const af::array& pos, const af::array& lhs,
                      const af::array& rhs) {
    if (pos.dims() != lhs.dims())
        throw make_error(
            "Position and lhs velocity must have the same dimensions");
    else if (lhs.dims() != rhs.dims())
        throw make_error(
            "Position and rhs velocity must have the same dimensions");
    else if (rhs.dims()[0] != 4)
        throw make_error("Arrays must have 4 principal coordinates");

    return af::matmul(af::moddims(lhs, 1, 4, lhs.dims()[1]), metric4(pos),
                      af::moddims(rhs, 4, 1, rhs.dims()[1]));
}

af::array norm4(const af::array& pos, const af::array& vel) {
    return dot_product(pos, vel, vel);
}

/**
 * @brief Computes the geodesics from the stablished metric, 4-vector positions
 * and velocities
 *
 * @param pos4
 * @param vel4
 * @return af::array
 */
af::array geodesics(const af::array& pos4, const af::array& vel4) {
    auto N = vel4.dims()[1];

    af::array uu = af::matmul(af::moddims(vel4, af::dim4(4, 1, N)),
                              af::moddims(vel4, af::dim4(1, 4, N)));
    uu           = af::moddims(uu, af::dim4(1, 4, 4, N));

    double diff = 0.001;

    af::array metric    = metric4(pos4);
    af::array invmetric = af::moddims(inv_metric(metric), af::dim4(4, 4, 1, N));

    af::array pos_diff = pos4 * diff;

    pos_diff = af::select(af::abs(pos_diff) > 1e-9, pos_diff, 1e-9);

    // Compute the partials of the metric with respect to coordinates indices
    af::array dt = af::constant(0, 4, 4, 1, N, f64);
    af::array dr =
        (metric4(pos4 +
                 pos_diff * af::array({0.0, 1., 0.0, 0.0}, af::dim4(4, 1))) -
         metric4(pos4 -
                 pos_diff * af::array({0.0, 1., 0.0, 0.0}, af::dim4(4, 1)))) /
        (af::moddims(pos_diff(1, af::span), af::dim4(1, 1, N)) * 2.0);
    af::array dtheta =
        (metric4(pos4 +
                 pos_diff * af::array({0.0, 0.0, 1., 0.0}, af::dim4(4, 1))) -
         metric4(pos4 -
                 pos_diff * af::array({0.0, 0.0, 1., 0.0}, af::dim4(4, 1)))) /
        (af::moddims(pos_diff(2, af::span), af::dim4(1, 1, N)) * 2.0);
    af::array dphi =
        (metric4(pos4 +
                 pos_diff * af::array({0.0, 0.0, 0.0, 1.}, af::dim4(4, 1))) -
         metric4(pos4 -
                 pos_diff * af::array({0.0, 0.0, 0.0, 1.}, af::dim4(4, 1)))) /
        (af::moddims(pos_diff(3, af::span), af::dim4(1, 1, N)) * 2.0);

    dr     = af::moddims(dr, af::dim4(4, 4, 1, N));
    dtheta = af::moddims(dtheta, af::dim4(4, 4, 1, N));
    dphi   = af::moddims(dphi, af::dim4(4, 4, 1, N));

    // Compute the einsum for each of the christoffel terms
    af::array partials = af::join(2, dt, dr, dtheta, dphi);
    af::array p1       = af::matmul(invmetric, partials);
    af::array p2       = af::reorder(p1, 0, 2, 1, 3);
    af::array p3 = af::matmul(invmetric, af::reorder(partials, 2, 0, 1, 3));

    auto christoffels = -0.5 * (p1 + p2 - p3);

    // Use the geodesics equation to find the 4-vector acceleration
    return af::moddims(af::sum(af::sum(christoffels * uu, 1), 2),
                       af::dim4(4, N));
}

/**
 * @brief Camera struct
 *
 * Contains all the data pertaining to the parameters for the image as seen from
 * the camera
 *
 */
struct Camera {
    af::array position;
    af::array lookat;
    double fov;
    double focal_length;
    uint32_t width;
    uint32_t height;

    af::array direction;
    af::array vertical;
    af::array horizontal;
    double aspect_ratio;

    Camera(const af::array& position_, const af::array& lookat_, double fov_,
           double focal_length_, uint32_t viewport_width_,
           uint32_t viewport_height_)
        : position(position_)
        , lookat(lookat_)
        , fov(fov_)
        , focal_length(focal_length_)
        , width(viewport_width_)
        , height(viewport_height_) {
        auto global_vertical = af::array(3, {0.0, 0.0, 1.0});

        // Compute the camera three main axes
        direction  = normalize3(lookat - position);
        horizontal = normalize3(cross_product(direction, global_vertical));
        vertical   = normalize3(cross_product(direction, horizontal));

        aspect_ratio = (double)width / (double)height;
    }

    /**
     * @brief Generates the initial rays 4-vector position and velocities
     * (direction) for the simulation
     *
     * @return std::pair<af::array, af::array> (pos4, vel4)
     */
    std::pair<af::array, af::array> generate_viewport_4rays() {
        auto& camera_direction  = direction;
        auto& camera_horizontal = horizontal;
        auto& camera_vertical   = vertical;
        auto& camera_position   = position;
        auto vfov               = fov;

        double viewport_height = 2.0 * focal_length * std::tan(vfov / 2.0);
        double viewport_width  = aspect_ratio * viewport_height;

        // Create rays in equally spaced directions of the viewport
        af::array viewport_rays = af::constant(0, 3, width, height, f64);
        viewport_rays +=
            (af::iota(af::dim4(1, width, 1), af::dim4(1, 1, height), f64) /
                 (width - 1) -
             0.5) *
            viewport_width * camera_horizontal;
        viewport_rays +=
            (af::iota(af::dim4(1, 1, height), af::dim4(1, width, 1), f64) /
                 (height - 1) -
             0.5) *
            viewport_height * camera_vertical;
        viewport_rays += focal_length * camera_direction;
        viewport_rays = af::moddims(af::reorder(viewport_rays, 1, 2, 0),
                                    af::dim4(width * height, 3))
                            .T();

        // Compute the initial position from which the rays are launched
        af::array viewport_position = viewport_rays + camera_position;
        af::array viewport_sph_pos  = cart_to_sph_position(viewport_position);

        // Normalize the ray directions
        viewport_rays = normalize3(viewport_rays);

        // Generate the position 4-vector
        auto camera_sph_pos = cart_to_sph_position(camera_position);
        af::array camera_pos4 =
            af::join(0, af::constant(0.0, 1, f64), camera_sph_pos);
        double camera_velocity =
            1.0 /
            af::sqrt(norm4(camera_pos4, af::array(4, {1.0, 0.0, 0.0, 0.0})))
                .scalar<double>();
        af::array camera_vel4 = af::array(4, {camera_velocity, 0.0, 0.0, 0.0});

        af::array viewport_rays_pos4 = af::join(
            0, af::constant(0.0, 1, width * height, f64), viewport_sph_pos);

        // Generate the velocity 4-vector by setting the camera to be stationary
        // with respect to an observer at infinity
        auto vv       = cart_to_sph_velocity(viewport_rays, viewport_position);
        af::array vvr = vv(0, af::span);
        af::array vvo = vv(1, af::span);
        af::array vvp = vv(2, af::span);
        auto viewport_sph_rays4 =
            af::join(0, af::constant(1, 1, width * height, f64), vvr, vvo, vvp);

        af::array dot = af::moddims(
            af::matmul(metric4(viewport_rays_pos4),
                       af::moddims(viewport_sph_rays4 * viewport_sph_rays4,
                                   af::dim4(4, 1, width * height))),
            af::dim4(4, width * height));

        // Normalize the 4-velocity vectors
        af::array viewport_vel =
            af::sqrt(-af::array(dot(0, af::span)) /
                     (dot(1, af::span) + dot(2, af::span) + dot(3, af::span)));
        af::array viewport_rays_vel4 =
            af::join(0, af::constant(camera_velocity, 1, width * height, f64),
                     vv * viewport_vel * camera_velocity);

        return {viewport_rays_pos4, viewport_rays_vel4};
    }
};

/**
 * @brief Object struct
 *
 * Contains the methods for testing if a ray has collided with the object
 *
 */
struct Object {
    using HasHit = af::array;
    using HitPos = af::array;

    /**
     * @brief Gets the color of the pixel that correspond to the ray that has
     * intersected with the object
     *
     * @param ray_begin begining
     * @param ray_end
     * @return af::array
     */
    virtual af::array get_color(const af::array& ray_begin,
                                const af::array& ray_end) const = 0;

    /**
     * @brief Returns a bool array if the rays have hit the object and the
     * correspoding position where the ray has hit
     *
     * @param ray_begin
     * @param ray_end
     * @return std::pair<HasHit, HitPos>
     */
    virtual std::pair<HasHit, HitPos> intersect(
        const af::array& ray_begin, const af::array& ray_end) const = 0;
};

struct AccretionDisk : public Object {
    af::array disk_color;
    af::array center;
    af::array normal;
    double radius;

    AccretionDisk(const af::array& center, const af::array& normal,
                  double radius)
        : disk_color(af::array(3, {209.f, 77.f, 0.f}))
        , center(center)
        , normal(normal)
        , radius(radius) {
        // disk_color = af::array(3, {254.f, 168.f, 29.f});
    }

    std::pair<HasHit, HitPos> intersect(
        const af::array& ray_begin, const af::array& ray_end) const override {
        uint32_t count = ray_begin.dims()[1];

        // Compute intersection of ray with a plane
        af::array has_hit = af::constant(0, count).as(b8);
        af::array hit_pos = ray_end;
        af::array a       = dot3(normal, center - ray_begin);
        af::array b       = dot3(normal, ray_end - ray_begin);
        af::array t       = af::select(b != 0.0, a / b, (double)0.0);

        af::array plane_intersect = (ray_end - ray_begin) * t + ray_begin;
        af::array dist            = norm3(plane_intersect - center);

        t = af::abs(t);

        // Determine if the intersection falls inside the disk radius and occurs
        // with the current ray segment
        has_hit = af::moddims((dist < radius) && (t <= 1.0) && (t > 0.0),
                              af::dim4(count));
        hit_pos = plane_intersect;

        return {has_hit, hit_pos};
    }

    af::array get_color(const af::array& ray_begin,
                        const af::array& ray_end) const override {
        auto pair       = intersect(ray_begin, ray_end);
        const auto& hit = pair.first;
        const auto& pos = pair.second;

        af::array color =
            disk_color.T() *
            af::exp(1.0f -
                    1.0f / af::pow(1.0 - (norm3(pos - center).T() - 1) / radius,
                                   2.0))
                .as(f32);

        color += af::randn(ray_begin.dims()[1]) * 5;

        return af::select(af::tile(hit, af::dim4(1, 3)), color, 0.f);
    }
};

/**
 * @brief Background struct
 *
 * Contains the methods for getting the color of background image
 *
 */
struct Background {
    af::array image;

    Background(const af::array& image_) { image = image_; }

    af::array get_color(const af::array& ray_begin,
                        const af::array& ray_end) const {
        auto dir           = ray_end - ray_begin;
        auto spherical_dir = cart_to_sph_position(dir);

        auto img_height = image.dims()[0];
        auto img_width  = image.dims()[1];
        auto count      = ray_begin.dims()[1];

        // Spherical mapping of the direction to a pixel of the image
        af::array o = spherical_dir(1, af::span);
        af::array p = spherical_dir(2, af::span);

        auto x = (p / af::Pi + 1.0) * img_width / 2.0;
        auto y = (o / af::Pi) * img_height;

        // Interpolate the colors of the image from the calculated pixel
        // positions
        af::array colors = af::approx2(image, af::moddims(y.as(f32), count),
                                       af::moddims(x.as(f32), count),
                                       af::interpType::AF_INTERP_CUBIC_SPLINE);

        // Zero out the color of any null rays
        colors = af::moddims(colors, af::dim4(count, 3));
        af::replace(colors, !af::isNaN(colors), 0.f);

        return colors;
    }
};

/**
 * @brief Transform the array of pixels to the correct image format to display
 *
 * @param image
 * @param width
 * @param height
 * @return af::array
 */
af::array rearrange_image(const af::array& image, uint32_t width,
                          uint32_t height) {
    return af::clamp(af::moddims(image, af::dim4(width, height, 3)).T(), 0.0,
                     255.0)
               .as(f32) /
           255.f;
}

/**
 * @brief Returns an rgb image containing the raytraced black hole from the
 * camera rays, spacetime metric, objects living in the space, and background
 *
 * @param initial_pos initial position from where the rays are launched
 * @param initial_vel initial velocities (directions) the rays have
 * @param objects the objects the rays can collide with
 * @param background the background of the scene
 * @param time how long are the rays traced through space
 * @param steps how many steps should be taken to trace the rays path
 * @param width width of the image the camera produces
 * @param height height of the image the camera produces
 * @param checks the intervals between steps to check if the rays have collided
 * with an object
 * @return af::array
 */
af::array generate_image(const af::array& initial_pos,
                         const af::array& initial_vel,
                         const std::vector<std::unique_ptr<Object> >& objects,
                         const Background& background, double time,
                         uint32_t steps, uint32_t width, uint32_t height,
                         uint32_t checks = 10) {
    uint32_t lines = initial_pos.dims()[1];

    auto def_step = time / (double)steps;
    auto dt       = af::constant(def_step, 1, lines, f64);

    auto result = af::constant(0, lines, 3, f32);

    auto pos = initial_pos;
    auto vel = initial_vel;

    af::Window window{(int)width, (int)height, "Black Hole Raytracing"};

    af::array bg_col;
    af::array begin_pos, end_pos;

    begin_pos = sph_to_cart_position3(pos(af::seq(1, 3), af::span));

    for (uint32_t i = 0; i < steps; ++i) {
        // Displays the current progress and approximate time needed to finish
        // it
        status_bar(i + 1, steps, "Progress:");

        // Stop rays entering the event horizon
        switch (scene) {
            case Scene::ROTATE_BH: {
                auto a = af::cos(pos(2, af::span)) * J / M;
                dt     = af::select(
                    pos(1, af::span) > 1.05 * (M + af::sqrt(M * M - a * a)), dt,
                    0.0);

                break;
            }

            case Scene::STATIC_BH: {
                dt = af::select(pos(1, af::span) > 2.05 * M, dt, 0.0);

                break;
            }

            default: break;
        }

        // RK4 method for second order differential equation
        auto dt2 = dt * dt;
        auto k1  = geodesics(pos, vel);
        auto k2  = geodesics(pos + vel * dt / 4.0 + k1 * dt2 / 32.0,
                             vel + k1 * dt / 4.0);
        auto k3  = geodesics(pos + vel * dt / 2.0 + (k1 + k2) * dt2 / 16.0,
                             vel + k2 * dt / 2.0);
        auto k4  = geodesics(pos + vel * dt + (k1 - k2 + k3 * 2.0) * dt2 / 4.0,
                             vel + (k1 - k2 * 2.0 + 2.0 * k3) * dt);

        pos += vel * dt + (k1 + k2 * 8.0 + k3 * 2.0 + k4) * dt2 / 24.0;
        vel += (k1 + k3 * 4.0 + k4) * dt / 6.0;

        // Update image
        if (i % checks == (checks - 1)) {
            end_pos = sph_to_cart_position3(pos(af::seq(1, 3), af::span));

            // Check if light ray intersect an object
            for (const auto& obj : objects)
                result += obj->get_color(begin_pos, end_pos);

            // Update background colors from rays
            bg_col = background.get_color(begin_pos, end_pos);

            // Display image
            window.image(rearrange_image(result + bg_col, width, height));

            begin_pos = end_pos;
        }
    }

    result += bg_col;

    return rearrange_image(result, width, height);
}

void raytracing(uint32_t width, uint32_t height) {
    // Set the parameters of the raytraced image
    double vfov         = radians(90.0);
    double focal_length = 0.01;

    // Set the parameters of the camera
    af::array global_vertical = af::array(3, {0.0, 0.0, 1.0});
    af::array camera_position = af::array(3, {-7.0, 6.0, 2.0});
    af::array camera_lookat   = af::array(3, {0.0, 0.0, 0.0});

    // Set the background of the scene
    auto bg_image =
        af::loadimage(ASSETS_DIR "/examples/images/westerlund.jpg", true);
    auto background = Background(bg_image);

    // Set the objects living in the scene
    std::vector<std::unique_ptr<Object> > objects;
    if (scene != Scene::WORMHOLE)
        objects.push_back(std::make_unique<AccretionDisk>(
            af::array(3, {0.0, 0.0, 0.0}), af::array(3, {0.0, 0.0, 1.0}),
            4.0 * M));

    // Generate rays from the camera
    auto camera = Camera(camera_position, camera_lookat, vfov, focal_length,
                         width, height);
    auto pair   = camera.generate_viewport_4rays();
    const auto& ray4_pos = pair.first;
    const auto& ray4_vel = pair.second;

    // Generate raytraced image
    auto image = generate_image(ray4_pos, ray4_vel, objects, background, 18,
                                1000, width, height, 4);

    // Save image
    af::saveImage("result.jpg", image);
}

int main(int argc, char** argv) {
    int device = argc > 1 ? std::atoi(argv[1]) : 0;

    int width  = argc > 2 ? std::atoi(argv[2]) : 100;
    int height = argc > 3 ? std::atoi(argv[3]) : 100;

    try {
        af::setDevice(device);
        af::info();

        std::cout << "** ArrayFire Black Hole Raytracing Demo\n\n";

        raytracing(width, height);
    } catch (const af::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}