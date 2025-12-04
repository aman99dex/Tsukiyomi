#ifndef UTILS_H_
#define UTILS_H_

#include <filesystem>

#include <torch/torch.h>

#include "renderer.h"

// Initialization functions
void set_seed(int seed);
torch::Device get_device();
bool parse_arguments(int argc, char *argv[], std::filesystem::path &data_path,
                     std::filesystem::path &output_path);

// File handling functions
std::vector<char> load_binary_file(const std::filesystem::path &file_path);
torch::Tensor load_tensor(const std::filesystem::path &file_path);
float load_focal(const std::filesystem::path &file_path);
void save_image(const torch::Tensor &tensor,
                const std::filesystem::path &file_path);

// Rendering helper functions
void render_and_save_orbit_views(const NeRFRenderer &renderer,
                                 int num_frames,
                                 const std::filesystem::path &output_folder,
                                 float radius = 4.0f,
                                 float start_distance = 2.0f,
                                 float end_distance = 6.0f, int n_samples = 64);

/*
void render_and_save_light_orbit(const NeRFRenderer &renderer, int num_frames,
                                 const std::filesystem::path &output_folder,
                                 float radius = 4.0f,
                                 float start_distance = 2.0f,
                                 float end_distance = 6.0f, int n_samples = 64);
*/

void render_dataset_views(const NeRFRenderer &renderer,
                          const torch::Tensor &poses,
                          const torch::Tensor &images,
                          const std::filesystem::path &output_path,
                          int step = 10);

void save_point_cloud(NeRFModel &model, const torch::Device &device,
                      const std::filesystem::path &output_path,
                      int resolution = 100, float threshold = 5.0f);

// Transformation and pose functions
torch::Tensor create_spherical_pose(float azimuth, float elevation,
                                    float radius, bool flip_axes = false);
torch::Tensor create_translation_matrix(float t);
torch::Tensor create_phi_rotation_matrix(float phi);
torch::Tensor create_theta_rotation_matrix(float theta);

// PBR Math Helpers
torch::Tensor schlick_fresnel(const torch::Tensor &F0, const torch::Tensor &cos_theta);
torch::Tensor smith_geometry(const torch::Tensor &N, const torch::Tensor &V, const torch::Tensor &L, const torch::Tensor &roughness);
torch::Tensor ggx_distribution(const torch::Tensor &N, const torch::Tensor &H, const torch::Tensor &roughness);

#endif // UTILS_H_
