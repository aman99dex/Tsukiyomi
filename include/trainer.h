#pragma once

#include "model.h"
#include "renderer.h"
#include "utils.h"
#include <torch/torch.h>
#include <filesystem>
#include <mutex>
#include <atomic>
#include <optional>

class Trainer {
public:
    Trainer(const std::filesystem::path& data_path, const std::filesystem::path& output_path, const torch::Device& device);
    
    // Training control
    void step();
    void save_checkpoint();
    
    // Async preview interface
    void request_preview(const torch::Tensor& pose, int H, int W, float near, float far);
    std::optional<torch::Tensor> get_preview();
    void process_preview(); // Called by training thread

    // Getters for GUI
    int get_iteration() const { return iteration_; }
    float get_loss() const { return current_loss_; }
    int get_preview_width() const { return preview_W_; }
    int get_preview_height() const { return preview_H_; }
    
    // Accessors
    NeRFModel& get_model() { return model_; }
    NeILFModel& get_neilf_model() { return neilf_model_; }
    NeRFRenderer& get_renderer() { return renderer_; }
    const torch::Tensor& get_poses() const { return poses_; }
    float get_focal() const { return focal_; }

private:
    torch::Tensor render_preview(const torch::Tensor& pose, int H, int W, float near, float far);

    const torch::Device& device_;
    std::filesystem::path output_path_;
    
    // Models
    NeRFModel model_;
    NeILFModel neilf_model_;
    
    // Renderer
    NeRFRenderer renderer_;

    // Optimization
    std::unique_ptr<torch::optim::Adam> optimizer_;
    
    // Data
    torch::Tensor images_;
    torch::Tensor poses_;
    float focal_;
    int H_, W_;
    
    // State
    int iteration_ = 0;
    float current_loss_ = 0.0f;
    
    // Thread safety
    std::mutex mutex_;
    
    // Preview state
    std::mutex preview_mutex_;
    bool preview_requested_ = false;
    torch::Tensor preview_pose_;
    int preview_H_, preview_W_;
    float preview_near_ = 2.0f;
    float preview_far_ = 6.0f;
    torch::Tensor preview_image_;
    bool preview_ready_ = false;
};
