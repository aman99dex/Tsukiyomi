#include "trainer.h"
#include <iostream>

Trainer::Trainer(const std::filesystem::path& data_path, const std::filesystem::path& output_path, const torch::Device& device)
    : device_(device), output_path_(output_path), model_(device), neilf_model_(device), 
      renderer_(&model_, &neilf_model_, 0, 0, 0.0f, device) { // Temp init for renderer
    
    // Load data
    images_ = load_tensor(data_path / "images.pt").to(device_);
    poses_ = load_tensor(data_path / "poses.pt").to(device_);
    focal_ = load_focal(data_path / "focal.pt");
    
    H_ = images_.size(1);
    W_ = images_.size(2);
    
    // Re-init renderer with correct dims
    renderer_ = NeRFRenderer(&model_, &neilf_model_, H_, W_, focal_, device_);
    
    // Optimizer
    std::vector<torch::Tensor> params = model_.parameters();
    auto neilf_params = neilf_model_.parameters();
    params.insert(params.end(), neilf_params.begin(), neilf_params.end());
    
    optimizer_ = std::make_unique<torch::optim::Adam>(params, torch::optim::AdamOptions(5e-4));
}

void Trainer::step() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    int batch_size = 2048;
    
    // Random ray batching
    int img_idx = torch::randint(0, poses_.size(0), {1}, torch::dtype(torch::kLong)).item<int>();
    auto target_pose = poses_[img_idx];
    auto target_img = images_[img_idx];

    // Get rays
    auto rays = renderer_.get_rays(target_pose);
    auto rays_o = std::get<0>(rays).reshape({-1, 3});
    auto rays_d = std::get<1>(rays).reshape({-1, 3});

    // Random sampling of pixels
    auto coords = torch::randint(0, H_ * W_, {batch_size},
                                 torch::dtype(torch::kLong).device(device_));
    auto rays_o_batch = rays_o.index_select(0, coords);
    auto rays_d_batch = rays_d.index_select(0, coords);
    auto target_rgb = target_img.reshape({-1, 3}).index_select(0, coords);

    optimizer_->zero_grad();

    auto result = renderer_.render_rays(std::make_tuple(rays_o_batch, rays_d_batch),
                                       true); // randomize=true
    
    auto rgb_pred = result["rgb"];
    auto normal = result["normal"];
    auto roughness = result["roughness"];
    auto energy_map = result["energy"];
    auto specular_map = result["specular"];

    auto mse_loss = torch::mse_loss(rgb_pred, target_rgb);
    
    // Regularization Losses
    auto view_dir = -torch::nn::functional::normalize(rays_d_batch, torch::nn::functional::NormalizeFuncOptions().dim(-1));
    auto normal_loss = torch::mean(torch::relu(-torch::sum(normal * view_dir, -1)));
    
    auto r = roughness + 1e-6f;
    auto roughness_loss = -torch::mean(r * torch::log(r) + (1 - r) * torch::log(1 - r));

    auto energy_loss = torch::mean(torch::relu(energy_map - 1.0f));
    auto specular_loss = torch::mean(specular_map * roughness);

    auto loss = mse_loss + 0.1f * normal_loss + 0.01f * roughness_loss + 0.1f * energy_loss + 0.05f * specular_loss;

    loss.backward();
    optimizer_->step();
    
    current_loss_ = loss.item<float>();
    iteration_++;
    
    if (iteration_ % 10 == 0) {
        std::cout << "Iteration: " << iteration_ << ", Loss: " << current_loss_ << std::endl;
    }

    // Save validation frame every 100 iterations
    if (iteration_ % 100 == 0) {
        std::cout << "Saving validation frame..." << std::endl;
        auto test_pose = poses_[0];
        // Render full resolution with reasonable samples
        auto result = renderer_.render(test_pose, false, 2.0f, 6.0f, 64);
        auto rgb = result["rgb"].detach().cpu();
        
        std::string filename = "frame_" + std::to_string(iteration_) + ".png";
        save_image(rgb, output_path_ / filename);
        std::cout << "Saved " << filename << std::endl;
    }
}

void Trainer::save_checkpoint() {
    std::lock_guard<std::mutex> lock(mutex_);
    torch::save(model_.get_seq(), output_path_ / "model.pt");
    torch::save(neilf_model_.get_seq(), output_path_ / "neilf_model.pt");
}

torch::Tensor Trainer::render_preview(const torch::Tensor& pose, int H, int W, float near, float far) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Create a temporary renderer for the requested resolution
    NeRFRenderer preview_renderer(&model_, &neilf_model_, H, W, focal_, device_);
    
    // Render with fewer samples for speed in GUI
    auto result = preview_renderer.render(pose, false, near, far, 32); 
    auto rgb = result["rgb"].detach().cpu();
    
    // Debug print
    std::cout << "Preview Rendered. Mean: " << rgb.mean().item<float>() << " Max: " << rgb.max().item<float>() << std::endl;
    
    return rgb;
}

void Trainer::request_preview(const torch::Tensor& pose, int H, int W, float near, float far) {
    std::lock_guard<std::mutex> lock(preview_mutex_);
    if (!preview_requested_) {
        preview_pose_ = pose;
        preview_H_ = H;
        preview_W_ = W;
        preview_near_ = near;
        preview_far_ = far;
        preview_requested_ = true;
    }
}

void Trainer::process_preview() {
    bool requested = false;
    torch::Tensor pose;
    int H, W;
    float near, far;
    
    {
        std::lock_guard<std::mutex> lock(preview_mutex_);
        if (preview_requested_) {
            requested = true;
            pose = preview_pose_;
            H = preview_H_;
            W = preview_W_;
            near = preview_near_;
            far = preview_far_;
        }
    }
    
    if (requested) {
        // Render without holding the preview mutex (but render_preview holds the main mutex)
        auto image = render_preview(pose, H, W, near, far);
        
        {
            std::lock_guard<std::mutex> lock(preview_mutex_);
            preview_image_ = image;
            preview_ready_ = true;
            preview_requested_ = false;
        }
    }
}

std::optional<torch::Tensor> Trainer::get_preview() {
    std::lock_guard<std::mutex> lock(preview_mutex_);
    if (preview_ready_) {
        preview_ready_ = false;
        return preview_image_;
    }
    return std::nullopt;
}
