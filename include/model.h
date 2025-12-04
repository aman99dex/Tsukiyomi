#ifndef MODEL_H_
#define MODEL_H_

#include <torch/torch.h>

class NeRFModel : public torch::nn::Module {
public:
  NeRFModel(const torch::Device &device = torch::kCPU, int L_embed = 6,
            int D = 8, int W = 256);

  torch::Tensor forward(const torch::Tensor &input);
  torch::Tensor add_positional_encoding(const torch::Tensor &x) const;
  torch::nn::Sequential get_seq() const { return model_; }

private:
  int L_embed_;
  torch::nn::Sequential model_;
  const torch::Device &device_;
};

class NeILFModel : public torch::nn::Module {
public:
  NeILFModel(const torch::Device &device = torch::kCPU, int L_embed_pos = 6,
             int L_embed_dir = 4, int D = 4, int W = 128);

  torch::Tensor forward(const torch::Tensor &pos, const torch::Tensor &dir);
  torch::Tensor add_positional_encoding(const torch::Tensor &x, int L_embed) const;
  torch::nn::Sequential get_seq() const { return model_; }

private:
  int L_embed_pos_;
  int L_embed_dir_;
  torch::nn::Sequential model_;
  const torch::Device &device_;
};

#endif // MODEL_H_
