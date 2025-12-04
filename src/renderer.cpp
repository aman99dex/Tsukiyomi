#include "renderer.h"

#include "utils.h"

using namespace torch::indexing;

NeRFRenderer::NeRFRenderer(NeRFModel *model, NeILFModel *neilf_model, int H, int W, float focal,
                           const torch::Device device)
    : model_(model), neilf_model_(neilf_model), H_(H), W_(W), focal_(focal), device_(device) {}

std::map<std::string, torch::Tensor> NeRFRenderer::render(const torch::Tensor &pose,
                                   bool randomize, float start_distance,
                                   float end_distance, int n_samples,
                                   int batch_size,
                                   const torch::Tensor &override_albedo,
                                   const torch::Tensor &override_roughness,
                                   const torch::Tensor &override_metallic) const {
  torch::NoGradGuard no_grad; // Disable gradient tracking for inference
  auto rays = get_rays(pose);
  auto rays_o = std::get<0>(rays).reshape({-1, 3});
  auto rays_d = std::get<1>(rays).reshape({-1, 3});

  // Shuffle rays during training
  torch::Tensor indices;
  if (randomize) {
    indices = torch::randperm(rays_o.size(0),
                              torch::dtype(torch::kLong).device(device_));
    rays_o = rays_o.index_select(0, indices);
    rays_d = rays_d.index_select(0, indices);
  }

  // Process rays in batches to avoid OOM
  std::vector<torch::Tensor> rgb_chunks;
  std::vector<torch::Tensor> albedo_chunks;
  std::vector<torch::Tensor> roughness_chunks;
  std::vector<torch::Tensor> normal_chunks;
  std::vector<torch::Tensor> depth_chunks;
  
  int n_rays = rays_o.size(0);
  int chunk_size = 4096; // Adjust based on memory
  
  for (int i = 0; i < n_rays; i += chunk_size) {
      int end = std::min(i + chunk_size, n_rays);
      auto chunk_o = rays_o.slice(0, i, end);
      auto chunk_d = rays_d.slice(0, i, end);
      auto chunk_rays = std::make_tuple(chunk_o, chunk_d);
      
      auto chunk_result = render_rays(chunk_rays, randomize, start_distance,
                                      end_distance, n_samples, batch_size,
                                      override_albedo, override_roughness, override_metallic);
      
      rgb_chunks.push_back(chunk_result["rgb"]);
      albedo_chunks.push_back(chunk_result["albedo"]);
      roughness_chunks.push_back(chunk_result["roughness"]);
      normal_chunks.push_back(chunk_result["normal"]);
      depth_chunks.push_back(chunk_result["depth"]);
  }
  
  auto rgb_flat = torch::cat(rgb_chunks, 0);
  auto albedo_flat = torch::cat(albedo_chunks, 0);
  auto roughness_flat = torch::cat(roughness_chunks, 0);
  auto normal_flat = torch::cat(normal_chunks, 0);
  auto depth_flat = torch::cat(depth_chunks, 0);

  // Unshuffle if randomized
  if (randomize) {
    auto inverse_indices = torch::argsort(indices);
    rgb_flat = rgb_flat.index_select(0, inverse_indices);
    albedo_flat = albedo_flat.index_select(0, inverse_indices);
    roughness_flat = roughness_flat.index_select(0, inverse_indices);
    normal_flat = normal_flat.index_select(0, inverse_indices);
    depth_flat = depth_flat.index_select(0, inverse_indices);
  }

  std::map<std::string, torch::Tensor> result;
  result["rgb"] = rgb_flat.reshape({H_, W_, 3});
  result["albedo"] = albedo_flat.reshape({H_, W_, 3});
  result["roughness"] = roughness_flat.reshape({H_, W_, 1});
  result["normal"] = normal_flat.reshape({H_, W_, 3});
  result["depth"] = depth_flat.reshape({H_, W_, 1});
  
  return result;
}

NeRFRenderer::RayData NeRFRenderer::get_rays(const torch::Tensor &pose) const {
  auto i = torch::linspace(0, W_ - 1, W_, torch::dtype(torch::kFloat32).device(device_));
  auto j = torch::linspace(0, H_ - 1, H_, torch::dtype(torch::kFloat32).device(device_));
  auto grid = torch::meshgrid({j, i}, "ij");
  auto u = grid[1];
  auto v = grid[0];

  auto dirs = torch::stack({(u - W_ * 0.5f) / focal_, -(v - H_ * 0.5f) / focal_,
                            -torch::ones_like(u)},
                           -1);
  auto rays_d = torch::sum(dirs.unsqueeze(-2) * pose.slice(0, 0, 3, 1).slice(1, 0, 3, 1), -1);
  auto rays_o = pose.slice(0, 0, 3, 1).slice(1, 3, 4, 1).reshape({1, 1, 3}).expand(rays_d.sizes());

  return std::make_tuple(rays_o, rays_d);
}

std::map<std::string, torch::Tensor> NeRFRenderer::render_rays(const RayData &rays,
                                        bool randomize, float start_distance,
                                        float end_distance, int n_samples,
                                        int batch_size,
                                        const torch::Tensor &override_albedo,
                                        const torch::Tensor &override_roughness,
                                        const torch::Tensor &override_metallic) const {
  // Unpack the ray origins and directions
  auto rays_o = std::get<0>(rays);
  auto rays_d = std::get<1>(rays);

  // Determine the number of rays (N)
  int N = rays_o.size(0);

  // Compute 3D query points
  auto z_vals =
      torch::linspace(start_distance, end_distance, n_samples, device_)
          .reshape({1, n_samples})
          .expand({N, n_samples})
          .clone();
  if (randomize) {
    z_vals += torch::rand({N, n_samples}, device_) *
              (start_distance - end_distance) / n_samples;
  }
  auto pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1);

  // Encode points
  auto pts_flat = pts.view({-1, 3});
  auto pts_embedded = model_->add_positional_encoding(pts_flat);

  // Batch-process points
  int n_pts = pts_flat.size(0);
  torch::Tensor raw;
  for (int i = 0; i < n_pts; i += batch_size) {
    auto batch = pts_embedded.slice(0, i, std::min(i + batch_size, n_pts));
    auto batch_raw = model_->forward(batch);
    if (i == 0) {
      raw = batch_raw;
    } else {
      raw = torch::cat({raw, batch_raw}, 0);
    }
  }
  raw = raw.view({N, n_samples, 9});

  // Get volume properties
  auto albedo = torch::sigmoid(raw.index({"...", Slice(0, 3)}));
  if (override_albedo.defined() && override_albedo.numel() > 0) {
      albedo = override_albedo.to(device_).expand_as(albedo);
  }

  auto roughness = torch::sigmoid(raw.index({"...", 3}));
  if (override_roughness.defined() && override_roughness.numel() > 0) {
      roughness = override_roughness.to(device_).expand_as(roughness);
  }

  auto metallic = torch::sigmoid(raw.index({"...", 4}));
  if (override_metallic.defined() && override_metallic.numel() > 0) {
      metallic = override_metallic.to(device_).expand_as(metallic);
  }
  auto normal = torch::nn::functional::normalize(
      torch::tanh(raw.index({"...", Slice(5, 8)})),
      torch::nn::functional::NormalizeFuncOptions().dim(-1));
  auto sigma_a = torch::nn::functional::softplus(raw.index({"...", 8}));

  // --- Monte Carlo Integration for PBR Shading ---
  int n_light_samples = 8; // Number of light samples per point (keep low for training speed)
  
  // View direction
  auto view_dir = torch::nn::functional::normalize(
      -rays_d.unsqueeze(1).expand_as(pts),
      torch::nn::functional::NormalizeFuncOptions().dim(-1));

  // Surface properties
  auto F0 = torch::tensor({0.04f, 0.04f, 0.04f}, device_).view({1, 1, 3}); 
  F0 = torch::lerp(F0, albedo, metallic.unsqueeze(-1));
  
  auto accumulated_radiance = torch::zeros_like(albedo);
  auto accumulated_energy = torch::zeros_like(albedo.index({"...", 0})).unsqueeze(-1); // [N, 1]
  auto accumulated_specular = torch::zeros_like(albedo);

  for (int l = 0; l < n_light_samples; ++l) {
      // 1. Sample random light direction (Uniform sphere sampling for now)
      // TODO: Importance sampling based on GGX would be better
      auto rand_theta = torch::rand_like(sigma_a) * 2 * M_PI;
      auto rand_phi = torch::acos(1 - 2 * torch::rand_like(sigma_a));
      
      auto lx = torch::sin(rand_phi) * torch::cos(rand_theta);
      auto ly = torch::sin(rand_phi) * torch::sin(rand_theta);
      auto lz = torch::cos(rand_phi);
      auto light_dir = torch::stack({lx, ly, lz}, -1); // [N, n_samples, 3]
      
      // 2. Query NeILF for incident radiance
      // NeILF takes (position, direction)
      // Flatten for batch processing
      auto pts_flat = pts.reshape({-1, 3});
      auto light_dir_flat = light_dir.reshape({-1, 3});
      
      // Batch processing for NeILF to avoid OOM
      torch::Tensor Li_flat;
      int n_neilf_pts = pts_flat.size(0);
      int neilf_batch_size = 65536; 
      
      for (int k = 0; k < n_neilf_pts; k += neilf_batch_size) {
          int end = std::min(k + neilf_batch_size, n_neilf_pts);
          auto pts_batch = pts_flat.slice(0, k, end);
          auto dir_batch = light_dir_flat.slice(0, k, end);
          auto Li_batch = neilf_model_->forward(pts_batch, dir_batch);
          
          if (k == 0) {
              Li_flat = Li_batch;
          } else {
              Li_flat = torch::cat({Li_flat, Li_batch}, 0);
          }
      }
      
      auto Li = Li_flat.reshape({pts.size(0), pts.size(1), 3});
      
      // 3. Compute BRDF
      auto half_vec = torch::nn::functional::normalize(
          light_dir + view_dir,
          torch::nn::functional::NormalizeFuncOptions().dim(-1));

      auto NdotL = torch::relu(torch::sum(normal * light_dir, -1, true));
      auto NdotV = torch::relu(torch::sum(normal * view_dir, -1, true));
      
      auto D = ggx_distribution(normal, half_vec, roughness.unsqueeze(-1));
      auto G = smith_geometry(normal, view_dir, light_dir, roughness.unsqueeze(-1));
      auto F = schlick_fresnel(F0, torch::relu(torch::sum(half_vec * view_dir, -1, true)));

      auto numerator = D * G * F;
      auto denominator = 4.0f * NdotV * NdotL + 1e-6f;
      auto specular = numerator / denominator;

      auto kS = F;
      auto kD = (1.0f - kS) * (1.0f - metallic.unsqueeze(-1));

      auto brdf = (kD * albedo / M_PI + specular);
      
      // 4. Accumulate: Li * BRDF * dot(N, L)
      // Note: PDF for uniform sphere sampling is 1/(4*PI).
      // Integral = Sum(Li * BRDF * NdotL) * (4*PI / N_samples)
      accumulated_radiance += Li * brdf * NdotL;
      
      // Accumulate Energy (Integral of BRDF * NdotL)
      // We sum the scalar contribution of the BRDF lobes
      // For energy conservation, we check if the total reflected energy <= 1
      // Approximate energy by summing the RGB channels or just taking the luminance
      auto energy_contribution = (brdf * NdotL).mean(-1, true); 
      accumulated_energy += energy_contribution;

      accumulated_specular += specular * NdotL;
  }
  
  // Normalize by number of samples and sphere area (4*PI)
  float norm_factor = 4.0f * M_PI / n_light_samples;
  auto rgb = accumulated_radiance * norm_factor;
  auto energy = accumulated_energy * norm_factor;
  auto specular_out = accumulated_specular * norm_factor;
  
  rgb = torch::clamp(rgb, 0.0f, 1.0f);

  // Render volume
  auto dists = torch::cat({z_vals.index({"...", Slice(1, None)}) -
                               z_vals.index({"...", Slice(None, -1)}),
                           torch::full({1}, 1e10, device_).expand({N, 1})},
                          -1);
  auto alpha = 1.0 - torch::exp(-sigma_a * dists);
  auto weights = torch::cumprod(1.0 - alpha + 1e-10, -1);
  weights = alpha * torch::cat({torch::ones({N, 1}, device_),
                                weights.index({"...", Slice(None, -1)})},
                               -1);

  auto rgb_map = torch::sum(weights.unsqueeze(-1) * rgb, -2);
  auto depth_map = torch::sum(weights * z_vals, -1).unsqueeze(-1);
  auto albedo_map = torch::sum(weights.unsqueeze(-1) * albedo, -2);
  auto roughness_map = torch::sum(weights.unsqueeze(-1) * roughness.unsqueeze(-1), -2); // Fixed shape mismatch
  auto normal_map = torch::sum(weights.unsqueeze(-1) * normal, -2);
  auto energy_map = torch::sum(weights.unsqueeze(-1) * energy, -2);
  auto specular_map = torch::sum(weights.unsqueeze(-1) * specular_out, -2);
  
  std::map<std::string, torch::Tensor> result;
  result["rgb"] = rgb_map;
  result["depth"] = depth_map;
  result["albedo"] = albedo_map;
  result["roughness"] = roughness_map;
  result["normal"] = normal_map;
  result["energy"] = energy_map;
  result["specular"] = specular_map;

  return result;
}
