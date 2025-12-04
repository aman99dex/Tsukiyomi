#include "model.h"

NeRFModel::NeRFModel(const torch::Device &device, int L_embed, int D, int W)
    : device_(device), L_embed_(L_embed) {
  // Create FFN
  auto input_dim = 3 + 3 * 2 * L_embed;
  model_->push_back(torch::nn::Linear(input_dim, W));
  model_->push_back(torch::nn::Functional(torch::relu));
  for (int i = 0; i < D - 2; i++) {
    model_->push_back(torch::nn::Linear(W, W));
    model_->push_back(torch::nn::Functional(torch::relu));
  }
  model_->push_back(torch::nn::Linear(W, 9));
  model_->to(device_);
  register_module("model", model_);

  this->to(device_);
}

torch::Tensor NeRFModel::forward(const torch::Tensor &input) {
  torch::Tensor x = model_->forward(input);
  return x;
}

torch::Tensor NeRFModel::add_positional_encoding(const torch::Tensor &x) const {
  std::vector<torch::Tensor> enc = {x};
  for (int i = 0; i < L_embed_; i++) {
    enc.push_back(torch::sin(std::pow(2.0f, i) * x));
    enc.push_back(torch::cos(std::pow(2.0f, i) * x));
  }
  return torch::cat(enc, -1);
}

NeILFModel::NeILFModel(const torch::Device &device, int L_embed_pos,
                       int L_embed_dir, int D, int W)
    : device_(device), L_embed_pos_(L_embed_pos), L_embed_dir_(L_embed_dir) {
  
  // Input dim: (3 + 3*2*L_pos) + (3 + 3*2*L_dir)
  auto input_dim = (3 + 3 * 2 * L_embed_pos) + (3 + 3 * 2 * L_embed_dir);
  
  model_->push_back(torch::nn::Linear(input_dim, W));
  model_->push_back(torch::nn::Functional(torch::relu));
  
  for (int i = 0; i < D - 2; i++) {
    model_->push_back(torch::nn::Linear(W, W));
    model_->push_back(torch::nn::Functional(torch::relu));
  }
  
  model_->push_back(torch::nn::Linear(W, 3)); // RGB output
  model_->push_back(torch::nn::Softplus()); // Radiance must be positive
  
  model_->to(device_);
  register_module("model", model_);
  
  this->to(device_);
}

torch::Tensor NeILFModel::forward(const torch::Tensor &pos, const torch::Tensor &dir) {
  auto pos_enc = add_positional_encoding(pos, L_embed_pos_);
  auto dir_enc = add_positional_encoding(dir, L_embed_dir_);
  auto x = torch::cat({pos_enc, dir_enc}, -1);
  return model_->forward(x);
}

torch::Tensor NeILFModel::add_positional_encoding(const torch::Tensor &x, int L_embed) const {
  std::vector<torch::Tensor> enc = {x};
  for (int i = 0; i < L_embed; i++) {
    enc.push_back(torch::sin(std::pow(2.0f, i) * x));
    enc.push_back(torch::cos(std::pow(2.0f, i) * x));
  }
  return torch::cat(enc, -1);
}
