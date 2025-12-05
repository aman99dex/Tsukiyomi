#include "utils.h"

#include <fstream>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace torch::indexing;

// Set up the random seed for reproducibility
void set_seed(int seed) {
  torch::manual_seed(seed);
  if (torch::cuda::is_available()) {
    torch::cuda::manual_seed(seed);
  }
}

// Determine the appropriate device for computation (CPU or GPU)
torch::Device get_device() {
  if (torch::cuda::is_available()) {
    std::cout << "Using CUDA device" << std::endl;
    return torch::kCUDA;
  }
  if (torch::mps::is_available()) {
    std::cout << "Using MPS device" << std::endl;
    return torch::kMPS;
  }
  std::cout << "Using CPU device" << std::endl;
  return torch::kCPU;
}

bool parse_arguments(int argc, char *argv[], std::filesystem::path &data_path,
                     std::filesystem::path &output_path) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <data_path> <output_path>"
              << std::endl;
    return false;
  }

  data_path = argv[1];
  output_path = argv[2];
  return true;
}

std::vector<char> load_binary_file(const std::filesystem::path &file_path) {
  std::ifstream input(file_path, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                          (std::istreambuf_iterator<char>()));
  input.close();
  return bytes;
}

torch::Tensor load_tensor(const std::filesystem::path &file_path) {
  std::vector<char> f = load_binary_file(file_path);
  torch::IValue x = torch::pickle_load(f);
  return x.toTensor();
}

float load_focal(const std::filesystem::path &file_path) {
  torch::Tensor focal_tensor = load_tensor(file_path);
  return focal_tensor.item<float>();
}

void save_image(const torch::Tensor &tensor,
                const std::filesystem::path &file_path) {
  // Assuming the input tensor is a 3-channel (HxWx3) image in the range [0, 1]
  auto height = tensor.size(0);
  auto width = tensor.size(1);
  auto max = tensor.max().item<float>();
  auto min = tensor.min().item<float>();
  
  std::cout << "Saving image " << file_path << " (Range: " << min << " to " << max << ")" << std::endl;

  torch::Tensor tensor_normalized;
  if (std::abs(max - min) < 1e-6) {
      tensor_normalized = torch::zeros_like(tensor).to(torch::kU8);
  } else {
      tensor_normalized = ((tensor - min) / (max - min))
                                   .mul(255)
                                   .clamp(0, 255)
                                   .to(torch::kU8);
  }
  
  tensor_normalized = tensor_normalized.to(torch::kCPU).flatten().contiguous();
  
  cv::Mat image(cv::Size(width, height), CV_8UC3, tensor_normalized.data_ptr());
  cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
  cv::imwrite(file_path.string(), image);
  cv::imwrite(file_path.string(), image);
}

void cleanup_old_previews(const std::filesystem::path &output_dir, int keep_count) {
    std::vector<std::pair<int, std::filesystem::path>> previews;
    
    // Iterate over files
    for (const auto& entry : std::filesystem::directory_iterator(output_dir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            // Check format: preview_<number>.png
            if (filename.rfind("preview_", 0) == 0 && filename.find(".png") != std::string::npos && filename != "preview_latest.png") {
                try {
                    // Extract number
                    std::string num_str = filename.substr(8, filename.length() - 12); // "preview_" len 8, ".png" len 4
                    int iter = std::stoi(num_str);
                    previews.push_back({iter, entry.path()});
                } catch (...) {
                    // Ignore malformed files
                }
            }
        }
    }
    
    // Sort descending by iteration
    std::sort(previews.begin(), previews.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    
    // Delete old ones
    if (previews.size() > keep_count) {
        for (size_t i = keep_count; i < previews.size(); ++i) {
            std::filesystem::remove(previews[i].second);
            // std::cout << "Cleaned up old preview: " << previews[i].second << std::endl;
        }
    }
}

void render_and_save_orbit_views(const NeRFRenderer &renderer,
                                 int num_frames,
                                 const std::filesystem::path &output_folder,
                                 float radius, float start_distance,
                                 float end_distance, int n_samples) {
  float elevation = -30.0f;

  for (int i = 0; i < num_frames; i++) {
    float azimuth = static_cast<float>(i) * 360.0f / num_frames;
    auto pose = create_spherical_pose(azimuth, elevation, radius).to(renderer.device());

    auto rendered_result = renderer.render(pose, false, start_distance,
                                          end_distance, n_samples);
    auto rendered_image = rendered_result["rgb"];

    std::string file_path =
        output_folder / ("frame_" + std::to_string(i) + ".png");
    save_image(rendered_image, file_path);
  }
}

/*
void render_and_save_light_orbit(const NeRFRenderer &renderer, int num_frames,
                                 const std::filesystem::path &output_folder,
                                 float radius, float start_distance,
                                 float end_distance, int n_samples) {
  // TODO: Implement environment rotation for NeILF
}
*/

void render_dataset_views(const NeRFRenderer &renderer,
                          const torch::Tensor &poses,
                          const torch::Tensor &images,
                          const std::filesystem::path &output_path,
                          int step) {
  std::cout << "Rendering dataset views..." << std::endl;
  int n_images = poses.size(0);
  for (int i = 0; i < n_images; i += step) {
    auto pose = poses[i];
    auto target = images[i];
    
    auto rendered_result = renderer.render(pose, false);
    auto rendered_image = rendered_result["rgb"];
    
    // Concatenate target and rendered image side-by-side
    auto combined = torch::cat({target, rendered_image}, 1);
    
    std::string file_path = output_path / ("dataset_view_" + std::to_string(i) + ".png");
    save_image(combined, file_path);
  }
}

void save_point_cloud(NeRFModel &model, const torch::Device &device,
                      const std::filesystem::path &output_path,
                      int resolution, float threshold) {
  std::cout << "Generating point cloud..." << std::endl;
  std::vector<std::string> lines;
  lines.push_back("ply");
  lines.push_back("format ascii 1.0");
  
  // Create a grid of points
  auto x = torch::linspace(-2.0, 2.0, resolution, torch::dtype(torch::kFloat32).device(device));
  auto y = torch::linspace(-2.0, 2.0, resolution, torch::dtype(torch::kFloat32).device(device));
  auto z = torch::linspace(-2.0, 2.0, resolution, torch::dtype(torch::kFloat32).device(device));
  auto grid = torch::meshgrid({x, y, z}, "xyz");
  auto pts = torch::stack({grid[0], grid[1], grid[2]}, -1).reshape({-1, 3});
  
  // Query model in batches
  int batch_size = 64000;
  int n_pts = pts.size(0);
  
  std::vector<std::tuple<float, float, float, int, int, int>> valid_points;
  
  for (int i = 0; i < n_pts; i += batch_size) {
    auto batch = pts.slice(0, i, std::min(i + batch_size, n_pts));
    auto batch_embedded = model.add_positional_encoding(batch);
    auto raw = model.forward(batch_embedded);
    
    auto sigma = torch::relu(raw.index({"...", 8}));
    auto rgb = torch::sigmoid(raw.index({"...", Slice(0, 3)}));
    
    auto mask = sigma > threshold;
    auto valid_indices = torch::nonzero(mask).squeeze();
    
    if (valid_indices.numel() > 0) {
        auto valid_pts = batch.index_select(0, valid_indices).cpu();
        auto valid_rgb = rgb.index_select(0, valid_indices).cpu();
        
        auto pts_acc = valid_pts.accessor<float, 2>();
        auto rgb_acc = valid_rgb.accessor<float, 2>();
        
        for (int j = 0; j < valid_pts.size(0); ++j) {
            valid_points.emplace_back(
                pts_acc[j][0], pts_acc[j][1], pts_acc[j][2],
                static_cast<int>(rgb_acc[j][0] * 255),
                static_cast<int>(rgb_acc[j][1] * 255),
                static_cast<int>(rgb_acc[j][2] * 255)
            );
        }
    }
  }
  
  lines.push_back("element vertex " + std::to_string(valid_points.size()));
  lines.push_back("property float x");
  lines.push_back("property float y");
  lines.push_back("property float z");
  lines.push_back("property uchar red");
  lines.push_back("property uchar green");
  lines.push_back("property uchar blue");
  lines.push_back("end_header");
  
  std::ofstream outfile(output_path);
  for (const auto& line : lines) {
      outfile << line << "\n";
  }
  
  for (const auto& p : valid_points) {
      outfile << std::get<0>(p) << " " << std::get<1>(p) << " " << std::get<2>(p) << " "
              << std::get<3>(p) << " " << std::get<4>(p) << " " << std::get<5>(p) << "\n";
  }
  outfile.close();
  std::cout << "Saved point cloud to " << output_path << std::endl;
}

torch::Tensor create_spherical_pose(float azimuth, float elevation,
                                    float radius, bool flip_axes) {
  float phi = elevation * (M_PI / 180.0f);
  float theta = azimuth * (M_PI / 180.0f);

  torch::Tensor c2w = create_translation_matrix(radius);
  c2w = create_phi_rotation_matrix(phi).matmul(c2w);
  c2w = create_theta_rotation_matrix(theta).matmul(c2w);
  
  if (flip_axes) {
      c2w = torch::tensor({{-1.0f, 0.0f, 0.0f, 0.0f},
                           {0.0f, 0.0f, 1.0f, 0.0f},
                           {0.0f, 1.0f, 0.0f, 0.0f},
                           {0.0f, 0.0f, 0.0f, 1.0f}})
                .matmul(c2w);
  }

  return c2w;
}

torch::Tensor create_translation_matrix(float t) {
  torch::Tensor t_mat = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f},
                                       {0.0f, 1.0f, 0.0f, 0.0f},
                                       {0.0f, 0.0f, 1.0f, t},
                                       {0.0f, 0.0f, 0.0f, 1.0f}});
  return t_mat;
}

torch::Tensor create_phi_rotation_matrix(float phi) {
  torch::Tensor phi_mat =
      torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f},
                     {0.0f, std::cos(phi), -std::sin(phi), 0.0f},
                     {0.0f, std::sin(phi), std::cos(phi), 0.0f},
                     {0.0f, 0.0f, 0.0f, 1.0f}});
  return phi_mat;
}

torch::Tensor create_theta_rotation_matrix(float theta) {
  torch::Tensor theta_mat =
      torch::tensor({{std::cos(theta), 0.0f, -std::sin(theta), 0.0f},
                     {0.0f, 1.0f, 0.0f, 0.0f},
                     {std::sin(theta), 0.0f, std::cos(theta), 0.0f},
                     {0.0f, 0.0f, 0.0f, 1.0f}});
  return theta_mat;
}

// PBR Math Helpers
torch::Tensor schlick_fresnel(const torch::Tensor &F0, const torch::Tensor &cos_theta) {
    return F0 + (1.0f - F0) * torch::pow(1.0f - cos_theta, 5.0f);
}

torch::Tensor smith_geometry(const torch::Tensor &N, const torch::Tensor &V, const torch::Tensor &L, const torch::Tensor &roughness) {
    auto k = torch::pow(roughness + 1.0f, 2.0f) / 8.0f;
    auto NdotV = torch::relu(torch::sum(N * V, -1, true));
    auto NdotL = torch::relu(torch::sum(N * L, -1, true));
    
    auto ggx1 = NdotV / (NdotV * (1.0f - k) + k + 1e-6f);
    auto ggx2 = NdotL / (NdotL * (1.0f - k) + k + 1e-6f);
    
    return ggx1 * ggx2;
}

torch::Tensor ggx_distribution(const torch::Tensor &N, const torch::Tensor &H, const torch::Tensor &roughness) {
    auto a2 = torch::pow(roughness, 4.0f);
    auto NdotH = torch::relu(torch::sum(N * H, -1, true));
    auto NdotH2 = torch::pow(NdotH, 2.0f);
    
    auto num = a2;
    auto denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
    denom = M_PI * torch::pow(denom, 2.0f) + 1e-6f;
    
    return num / denom;
}
