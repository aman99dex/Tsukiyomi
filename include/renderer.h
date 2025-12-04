#ifndef RENDERER_H_
#define RENDERER_H_

#include <torch/torch.h>
#include <map>
#include <string>
#include <tuple>

#include "model.h"

class NeRFRenderer {
public:
  NeRFRenderer(NeRFModel *model, NeILFModel *neilf_model, int H, int W, float focal,
               const torch::Device device);

  // Render a full image
  std::map<std::string, torch::Tensor> render(const torch::Tensor &pose,
                       bool randomize = false, float start_distance = 2.0f,
                       float end_distance = 5.0f, int n_samples = 64,
                       int batch_size = 64000,
                       const torch::Tensor &override_albedo = torch::Tensor(),
                       const torch::Tensor &override_roughness = torch::Tensor(),
                       const torch::Tensor &override_metallic = torch::Tensor()) const;

  const torch::Device& device() const { return device_; }

  using RayData = std::tuple<torch::Tensor, torch::Tensor>;

  RayData get_rays(const torch::Tensor &pose) const;
  
  // Render a batch of rays
  std::map<std::string, torch::Tensor> render_rays(const RayData &rays,
                            bool randomize = false, float start_distance = 2.0f,
                            float end_distance = 5.0f, int n_samples = 64,
                            int batch_size = 64000,
                            const torch::Tensor &override_albedo = torch::Tensor(),
                            const torch::Tensor &override_roughness = torch::Tensor(),
                            const torch::Tensor &override_metallic = torch::Tensor()) const;

private:
  NeRFModel *model_;
  NeILFModel *neilf_model_;
  torch::Device device_;
  int H_;
  int W_;
  float focal_;
};

#endif // RENDERER_H_
