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
    auto mse_loss = torch::mse_loss(rgb, batch_target);
    
    // Regularization (Optional - can be tuned)
    // For now, we rely on the PBR model to disentangle, but we can add smoothness here.
    auto loss = mse_loss;

    // Perform backward pass and update model parameters
    loss.backward();
    optimizer.step();

    // Log progress periodically
    // Log progress every iteration
    if (i % 10 == 0) {
      std::cout << "Iteration: " << i + 1 << " Loss: " << loss.item<float>()
                << std::endl;
    }

    // Render and save orbiting preview periodically
    if (i % plot_freq == 0) {
      // Render and save orbiting preview periodically
      std::cout << "Rendering preview..." << std::endl;
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

  // Material Override Demo
  std::cout << "Generating Material Override Demo..." << std::endl;
  
  // Gold Material
  auto gold_albedo = torch::tensor({1.0f, 0.766f, 0.336f}, device).view({1, 1, 3});
  auto gold_roughness = torch::tensor({0.2f}, device).view({1, 1, 1});
  auto gold_metallic = torch::tensor({1.0f}, device).view({1, 1, 1});
  
  // Plastic Material (Blue)
  auto plastic_albedo = torch::tensor({0.1f, 0.1f, 0.9f}, device).view({1, 1, 3});
  auto plastic_roughness = torch::tensor({0.5f}, device).view({1, 1, 1});
  auto plastic_metallic = torch::tensor({0.0f}, device).view({1, 1, 1});

  // Render a single view with overrides
  auto pose = poses[0]; // Use first pose
  
  auto gold_img = renderer_hd.render(pose, light_pos, false, 2.0f, 5.0f, 64, 64000, gold_albedo, gold_roughness, gold_metallic);
  save_image(gold_img, output_path / "material_gold.png");
  
  auto plastic_img = renderer_hd.render(pose, light_pos, false, 2.0f, 5.0f, 64, 64000, plastic_albedo, plastic_roughness, plastic_metallic);
  save_image(plastic_img, output_path / "material_plastic.png");

  // Render dataset views for comparison
  render_dataset_views(renderer_hd, poses, images, light_pos, output_path);

  // Save 3D point cloud
  save_point_cloud(model, device, output_path / "model.ply");

  return 0;
}
