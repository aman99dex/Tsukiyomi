#include "model.h"
#include "renderer.h"
#include "utils.h"

constexpr int seed = 1;
constexpr int n_iters = 10000;
constexpr int plot_freq = 50;
constexpr int n_preview_frames = 5;
constexpr int n_final_frames = 35;

int main(int argc, char *argv[]) {
  // Parse command-line arguments
  std::filesystem::path data_path;
  std::filesystem::path output_path;
  if (!parse_arguments(argc, argv, data_path, output_path)) {
    return 1;
  }

  // Set the random seed
  set_seed(seed);

  // Set number of threads for CPU parallelism
  int num_threads = std::thread::hardware_concurrency();
  if (num_threads > 0) {
      torch::set_num_threads(num_threads);
      std::cout << "Setting CPU threads to: " << num_threads << std::endl;
  }

  // Determine device for computation
  torch::Device device = get_device();

  // Define light position for PBR shading
  torch::Tensor light_pos = torch::tensor({2.0f, 2.0f, 2.0f}, device);

  // Load data: images, poses, and focal length
  torch::Tensor images = load_tensor(data_path / "images.pt").to(device);
  torch::Tensor poses = load_tensor(data_path / "poses.pt").to(device);
  float focal = load_focal(data_path / "focal.pt");

  // Display information about the loaded data
  std::cout << "Images: " << images.sizes() << std::endl;
  std::cout << "Poses: " << poses.sizes() << std::endl;
  std::cout << "Focal length: " << focal << std::endl;

  // Create NeRF model and renderer
  NeRFModel model(device);
  NeRFRenderer renderer(model, images.size(1), images.size(2), focal, device);

  // Set up the optimizer
  torch::optim::Adam optimizer(model.parameters(),
                               torch::optim::AdamOptions(5e-4));

  // Train the NeRF model
  for (int i = 0; i < n_iters; i++) {
    // Sample a random image and its corresponding pose
    int img_i = std::rand() % images.size(0);
    auto target = images[img_i];
    auto pose = poses[img_i];

    // Random ray sampling (Batching)
    auto rays = renderer.get_rays(pose.to(device));
    auto rays_o = std::get<0>(rays).view({-1, 3});
    auto rays_d = std::get<1>(rays).view({-1, 3});
    auto target_flat = target.view({-1, 3});

    // Select random indices
    auto indices = torch::randperm(rays_o.size(0), torch::dtype(torch::kLong).device(device)).slice(0, 0, 2048);
    auto batch_rays_o = rays_o.index_select(0, indices);
    auto batch_rays_d = rays_d.index_select(0, indices);
    auto batch_target = target_flat.index_select(0, indices);

    // Perform forward pass and compute loss
    optimizer.zero_grad();
    auto rgb = renderer.render_rays(std::make_tuple(batch_rays_o, batch_rays_d), light_pos, true);
    auto loss = torch::mse_loss(rgb, batch_target);

    // Perform backward pass and update model parameters
    loss.backward();
    optimizer.step();

    // Log progress periodically
    // Log progress every iteration
    if (i % 1 == 0) {
      std::cout << "Iteration: " << i + 1 << " Loss: " << loss.item<float>()
                << std::endl;
    }

    // Render and save orbiting preview periodically
    if (i % plot_freq == 0) {
      // Render and save orbiting preview periodically
      std::cout << "Rendering preview..." << std::endl;
      // For preview, we still want to render full images, so we use the original render method (which calls render_rays internally)
      // But we need to make sure render_rays can handle the full image shape (H*W rays)
      // The updated render_rays handles flat rays, so we just need to reshape the output back to HxW
      render_and_save_orbit_views(renderer, light_pos, n_preview_frames, output_path,
                                  4.0f);
    }
  }

  std::cout << "Done" << std::endl;

  // Generate high-resolution rendering using the trained model
  torch::NoGradGuard no_grad;
  NeRFRenderer renderer_hd(model, 300, 300, focal, device);
  render_and_save_orbit_views(renderer_hd, light_pos, n_final_frames, output_path, 2.1f,
                              0.8f, 3.2f);

  // Generate relighting demo (light orbiting around fixed object)
  std::cout << "Generating relighting demo..." << std::endl;
  render_and_save_light_orbit(renderer_hd, n_final_frames, output_path, 2.1f,
                              2.0f, 5.0f);

  return 0;
}
