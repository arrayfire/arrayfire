/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/DependencyModule.hpp>

#include <glad/glad.h>

#include <forge.h>

namespace arrayfire {
namespace common {

class ForgeModule : public DependencyModule {
   public:
    ForgeModule();

    MODULE_MEMBER(fg_create_window);
    MODULE_MEMBER(fg_get_window_context_handle);
    MODULE_MEMBER(fg_get_window_display_handle);
    MODULE_MEMBER(fg_make_window_current);
    MODULE_MEMBER(fg_set_window_font);
    MODULE_MEMBER(fg_set_window_position);
    MODULE_MEMBER(fg_set_window_title);
    MODULE_MEMBER(fg_set_window_size);
    MODULE_MEMBER(fg_set_window_colormap);
    MODULE_MEMBER(fg_draw_chart_to_cell);
    MODULE_MEMBER(fg_draw_chart);
    MODULE_MEMBER(fg_draw_image_to_cell);
    MODULE_MEMBER(fg_draw_image);
    MODULE_MEMBER(fg_swap_window_buffers);
    MODULE_MEMBER(fg_close_window);
    MODULE_MEMBER(fg_show_window);
    MODULE_MEMBER(fg_hide_window);
    MODULE_MEMBER(fg_release_window);

    MODULE_MEMBER(fg_create_font);
    MODULE_MEMBER(fg_load_system_font);
    MODULE_MEMBER(fg_release_font);

    MODULE_MEMBER(fg_create_image);
    MODULE_MEMBER(fg_get_pixel_buffer);
    MODULE_MEMBER(fg_get_image_size);
    MODULE_MEMBER(fg_release_image);

    MODULE_MEMBER(fg_create_plot);
    MODULE_MEMBER(fg_set_plot_color);
    MODULE_MEMBER(fg_get_plot_vertex_buffer);
    MODULE_MEMBER(fg_get_plot_vertex_buffer_size);
    MODULE_MEMBER(fg_release_plot);

    MODULE_MEMBER(fg_create_histogram);
    MODULE_MEMBER(fg_set_histogram_color);
    MODULE_MEMBER(fg_get_histogram_vertex_buffer);
    MODULE_MEMBER(fg_get_histogram_vertex_buffer_size);
    MODULE_MEMBER(fg_release_histogram);

    MODULE_MEMBER(fg_create_surface);
    MODULE_MEMBER(fg_set_surface_color);
    MODULE_MEMBER(fg_get_surface_vertex_buffer);
    MODULE_MEMBER(fg_get_surface_vertex_buffer_size);
    MODULE_MEMBER(fg_release_surface);

    MODULE_MEMBER(fg_create_vector_field);
    MODULE_MEMBER(fg_set_vector_field_color);
    MODULE_MEMBER(fg_get_vector_field_vertex_buffer_size);
    MODULE_MEMBER(fg_get_vector_field_direction_buffer_size);
    MODULE_MEMBER(fg_get_vector_field_vertex_buffer);
    MODULE_MEMBER(fg_get_vector_field_direction_buffer);
    MODULE_MEMBER(fg_release_vector_field);

    MODULE_MEMBER(fg_create_chart);
    MODULE_MEMBER(fg_get_chart_type);
    MODULE_MEMBER(fg_get_chart_axes_limits);
    MODULE_MEMBER(fg_set_chart_axes_limits);
    MODULE_MEMBER(fg_set_chart_axes_titles);
    MODULE_MEMBER(fg_set_chart_label_format);
    MODULE_MEMBER(fg_append_image_to_chart);
    MODULE_MEMBER(fg_append_plot_to_chart);
    MODULE_MEMBER(fg_append_histogram_to_chart);
    MODULE_MEMBER(fg_append_surface_to_chart);
    MODULE_MEMBER(fg_append_vector_field_to_chart);
    MODULE_MEMBER(fg_release_chart);

    MODULE_MEMBER(fg_err_to_string);
};

ForgeModule& forgePlugin();

#define FG_CHECK(fn)                                        \
    do {                                                    \
        fg_err e = (fn);                                    \
        if (e != FG_ERR_NONE) {                             \
            AF_ERROR("forge call failed", AF_ERR_INTERNAL); \
        }                                                   \
    } while (0);

}  // namespace common
}  // namespace arrayfire
