#include "renderer.h"

#include "utils.h"

using namespace torch::indexing;

NeRFRenderer::NeRFRenderer(NeRFModel &model, int H, int W, float focal,
                           const torch::Device &device)
    : model_(model), H_(H), W_(W), focal_(focal), device_(device) {}

torch::Tensor NeRFRenderer::render(const torch::Tensor &pose, const torch::Tensor &light_pos,
                                   bool randomize, float start_distance,
                                   float end_distance, int n_samples,
                                   int batch_size) const {
  auto rays = get_rays(pose.to(device_));
  return render_rays(rays, light_pos.to(device_), randomize, start_distance, end_distance, n_samples,
                     batch_size);
}

NeRFRenderer::RayData NeRFRenderer::get_rays(const torch::Tensor &pose) const {
  // Generate pixel indices along image width (i) and height (j)
  auto i = torch::arange(W_, torch::dtype(torch::kFloat32)).to(device_);
  auto j = torch::arange(H_, torch::dtype(torch::kFloat32)).to(device_);
  auto grid = torch::meshgrid({i, j}, "xy");
  auto ii = grid[0];
  auto jj = grid[1];

  // Compute the direction vector for each pixel in the image plane
  auto dirs = torch::stack({(ii - W_ * 0.5) / focal_, -(jj - H_ * 0.5) / focal_,
                            -torch::ones_like(ii)},
                           -1);

  // Transform the direction vectors from the camera's local coordinate system
  // to the global coordinate system
  auto rays_d = torch::sum(dirs.index({"...", None, Slice()}) *
                               pose.index({Slice(0, 3), Slice(0, 3)}),
                           -1);
  // Get the origin of the rays from the pose
  auto rays_o = pose.index({Slice(0, 3), -1}).expand(rays_d.sizes());

  return std::make_tuple(rays_o, rays_d);
}

torch::Tensor NeRFRenderer::render_rays(const RayData &rays, const torch::Tensor &light_pos,
                                        bool randomize, float start_distance,
                                        float end_distance, int n_samples,
                                        int batch_size) const {
  // Unpack the ray origins and directions
  auto rays_o = std::get<0>(rays);
  auto rays_d = std::get<1>(rays);

  // Compute 3D query points
  auto z_vals =
      torch::linspace(start_distance, end_distance, n_samples, device_)
          .reshape({1, 1, n_samples})
          .expand({H_, W_, n_samples})
          .clone();
  if (randomize) {
    z_vals += torch::rand({H_, W_, n_samples}, device_) *
              (start_distance - end_distance) / n_samples;
  }
  auto pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1);

  // Encode points
  auto pts_flat = pts.view({-1, 3});
  auto pts_embedded = model_.add_positional_encoding(pts_flat);

  // Batch-process points
  int n_pts = pts_flat.size(0);
  torch::Tensor raw;
  for (int i = 0; i < n_pts; i += batch_size) {
    auto batch = pts_embedded.slice(0, i, std::min(i + batch_size, n_pts));
    auto batch_raw = model_.forward(batch);
    if (i == 0) {
      raw = batch_raw;
    } else {
      raw = torch::cat({raw, batch_raw}, 0);
    }
  }
  raw = raw.view({H_, W_, n_samples, 8});

  // Get volume properties
  auto albedo = torch::sigmoid(raw.index({"...", Slice(0, 3)}));
  auto roughness = torch::sigmoid(raw.index({"...", 3}));
  auto normal = torch::nn::functional::normalize(
      torch::tanh(raw.index({"...", Slice(4, 7)})),
      torch::nn::functional::NormalizeFuncOptions().dim(-1));
  auto sigma_a = torch::relu(raw.index({"...", 7}));

  // PBR Shading (Blinn-Phong)
  auto light_dir = torch::nn::functional::normalize(
      light_pos - pts, torch::nn::functional::NormalizeFuncOptions().dim(-1));
  auto view_dir = torch::nn::functional::normalize(
      -rays_d.unsqueeze(2).expand_as(pts),
      torch::nn::functional::NormalizeFuncOptions().dim(-1));

  // Diffuse
  auto NdotL = torch::relu(torch::sum(normal * light_dir, -1));
  auto diffuse = albedo * NdotL.unsqueeze(-1);

  // Specular
  auto half_vec = torch::nn::functional::normalize(
      light_dir + view_dir,
      torch::nn::functional::NormalizeFuncOptions().dim(-1));
  auto NdotH = torch::relu(torch::sum(normal * half_vec, -1));
  auto specular = torch::pow(NdotH, 1.0f / (roughness + 1e-6f));

  auto rgb = diffuse + specular.unsqueeze(-1);

  // Render volume
  auto dists = torch::cat({z_vals.index({"...", Slice(1, None)}) -
                               z_vals.index({"...", Slice(None, -1)}),
                           torch::full({1}, 1e10, device_).expand({H_, W_, 1})},
                          -1);
  auto alpha = 1.0 - torch::exp(-sigma_a * dists);
  auto weights = torch::cumprod(1.0 - alpha + 1e-10, -1);
  weights = alpha * torch::cat({torch::ones({H_, W_, 1}, device_),
                                weights.index({"...", Slice(None, -1)})},
                               -1);

  auto rgb_map = torch::sum(weights.unsqueeze(-1) * rgb, -2);
  auto depth_map = torch::sum(weights * z_vals, -1);
  auto acc_map = torch::sum(weights, -1);

  return rgb_map;
}
