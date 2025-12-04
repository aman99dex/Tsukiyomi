#define GL_SILENCE_DEPRECATION
#include "trainer.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <thread>
#include <atomic>

// Global state for camera control
float cam_azimuth = 0.0f;
float cam_elevation = -30.0f;
float cam_radius = 4.0f;
std::atomic<bool> is_training(true);

// Helper to update texture
// Helper to update texture
void update_texture(GLuint texture_id, const torch::Tensor& image) {
    int width = image.size(1);
    int height = image.size(0);
    
    // Ensure CPU
    auto cpu_image = image.cpu();
    
    // Convert to RGBA (append alpha=1)
    auto alpha = torch::ones({height, width, 1}, torch::dtype(torch::kFloat32));
    auto rgba_image = torch::cat({cpu_image, alpha}, 2);
    
    // Convert to uint8 and ensure contiguous
    auto image_u8 = (rgba_image * 255).clamp(0, 255).to(torch::kU8).contiguous();
    
    // Debug: Print first pixel
    uint8_t* ptr = image_u8.data_ptr<uint8_t>();
    // std::cout << "Texture Update: " << width << "x" << height << " First Pixel: " 
    //           << (int)ptr[0] << "," << (int)ptr[1] << "," << (int)ptr[2] << std::endl;
    
    glBindTexture(GL_TEXTURE_2D, texture_id);
    
    // Check for errors
    while (glGetError() != GL_NO_ERROR); 
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, ptr);
    
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL Error in update_texture: " << err << std::endl;
    }
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0); // Unbind
}

int main(int argc, char *argv[]) {
    std::filesystem::path data_path;
    std::filesystem::path output_path;
    if (!parse_arguments(argc, argv, data_path, output_path)) {
        return 1;
    }
    
    // Ensure output directory exists
    std::filesystem::create_directories(output_path);

    // Initialize Trainer
    std::cout << "Initializing Trainer..." << std::endl;
    torch::Device device = get_device();
    Trainer trainer(data_path, output_path, device);
    std::cout << "Trainer initialized." << std::endl;
    
    // Start training thread
    std::atomic<bool> stop_training(false);
    std::thread training_thread([&]() {
        try {
            while (!stop_training) {
                if (is_training) {
                    trainer.step();
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                // Check for preview requests
                trainer.process_preview();
            }
        } catch (const std::exception& e) {
            std::cerr << "Training thread error: " << e.what() << std::endl;
        }
    });

    // Initialize GLFW
    std::cout << "Initializing GLFW..." << std::endl;
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return 1;
    }
    
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "PBR-NeRF Interactive", NULL, NULL);
    if (window == NULL) return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Texture for viewport
    GLuint texture_id;
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    unsigned char dummy_pixel[3] = {0, 0, 0};
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, dummy_pixel);
    
    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        static int current_preview_w = 200;
        static int current_preview_h = 200;

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        // Dockspace
        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());

        // 1. Control Panel
        ImGui::Begin("Controls");
        ImGui::Text("Iteration: %d", trainer.get_iteration());
        ImGui::Text("Loss: %.6f", trainer.get_loss());
        
        if (ImGui::Button(is_training ? "Pause Training" : "Resume Training")) {
            is_training = !is_training;
        }
        
        static int target_iterations = 50000;
        ImGui::InputInt("Target Iterations", &target_iterations);
        
        if (is_training && trainer.get_iteration() >= target_iterations) {
            is_training = false;
        }
        
        if (ImGui::Button("Save Checkpoint")) {
            trainer.save_checkpoint();
        }
        
        ImGui::Separator();
        ImGui::Text("Camera");
        ImGui::SliderFloat("Azimuth", &cam_azimuth, 0.0f, 360.0f);
        ImGui::SliderFloat("Elevation", &cam_elevation, -89.0f, 89.0f);
        ImGui::SliderFloat("Radius", &cam_radius, 0.1f, 10.0f); // Allow closer zoom
        
        static float near_plane = 2.0f;
        static float far_plane = 6.0f;
        ImGui::SliderFloat("Near Plane", &near_plane, 0.1f, 10.0f);
        ImGui::SliderFloat("Far Plane", &far_plane, 1.0f, 20.0f);
        
        static bool flip_axes = true;
        ImGui::Checkbox("Flip Axes (Blender/NeRF)", &flip_axes);
        
        ImGui::End();

        // 2. Viewport
        ImGui::SetNextWindowSize(ImVec2(400, 400), ImGuiCond_FirstUseEver);
        ImGui::Begin("Viewport");
        
        // Calculate pose
        auto pose = create_spherical_pose(cam_azimuth, cam_elevation, cam_radius, flip_axes).to(device);
        
        // Display image
        ImVec2 avail_size = ImGui::GetContentRegionAvail();
        if (avail_size.x < 10) avail_size.x = 200; // Fallback
        if (avail_size.y < 10) avail_size.y = 200; // Fallback
        
        // Maintain aspect ratio
        float aspect = (float)current_preview_w / (float)current_preview_h;
        if (aspect > 0) {
            if (avail_size.x / avail_size.y > aspect) {
                avail_size.x = avail_size.y * aspect;
            } else {
                avail_size.y = avail_size.x / aspect;
            }
        }

        // Draw Image with flipped UVs (0,1) -> (1,0) for OpenGL
        ImGui::Image((void*)(intptr_t)texture_id, avail_size, ImVec2(0, 1), ImVec2(1, 0), ImVec4(1,1,1,1), ImVec4(1,1,1,0.5));
        
        // Overlay Info
        ImGui::SetCursorPos(ImVec2(10, 10));
        ImGui::TextColored(ImVec4(0, 1, 0, 1), "Preview: %dx%d", current_preview_w, current_preview_h);
        
        // Mouse Controls
        if (ImGui::IsItemHovered()) {
            // Zoom (Scroll)
            float wheel = ImGui::GetIO().MouseWheel;
            if (wheel != 0.0f) {
                cam_radius -= wheel * 0.5f;
                cam_radius = std::max(0.1f, std::min(cam_radius, 10.0f));
            }
            
            // Rotate (Drag Left Mouse)
            if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                ImVec2 drag = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
                cam_azimuth -= drag.x * 0.5f;
                cam_elevation += drag.y * 0.5f;
                
                // Clamp/Wrap
                if (cam_azimuth > 360.0f) cam_azimuth -= 360.0f;
                if (cam_azimuth < 0.0f) cam_azimuth += 360.0f;
                cam_elevation = std::max(-89.0f, std::min(cam_elevation, 89.0f));
                
                ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
            }
        }
        
        ImGui::End();

        // 3. Status Bar
        ImGui::Begin("Status");
        if (is_training) {
            ImGui::TextColored(ImVec4(0, 1, 0, 1), "Training Running...");
        } else {
            ImGui::TextColored(ImVec4(1, 1, 0, 1), "Training Paused");
        }
        ImGui::SameLine();
        ImGui::Text("| Iter: %d | Loss: %.6f | FPS: %.1f", trainer.get_iteration(), trainer.get_loss(), ImGui::GetIO().Framerate);
        ImGui::End();

        // Request preview update
        // 1. Update on interaction (time-based throttle)
        // 2. Update on training progress (iteration-based)
        static double last_request_time = 0.0;
        // Request preview update
        // 1. Update on interaction (time-based throttle)
        // 2. Update on training progress (iteration-based)
        // static double last_request_time = 0.0; // Moved to top
        double current_time = glfwGetTime();
        
        // Simple 1Hz timer for preview
        if (current_time - last_request_time > 1.0) {
            trainer.request_preview(pose, 200, 200, near_plane, far_plane);
            last_request_time = current_time;
        }
        
        // Check if new preview is available
        auto new_preview = trainer.get_preview();
        if (new_preview.has_value()) {
            std::cout << "Preview updated! Saving to disk..." << std::endl;
            auto img = new_preview.value();
            update_texture(texture_id, img);
            
            // Update dimensions for aspect ratio
            current_preview_w = img.size(1);
            current_preview_h = img.size(0);
            
            // Save to disk
            std::string filename = "preview_" + std::to_string(trainer.get_iteration()) + ".png";
            save_image(img, output_path / filename);
            save_image(img, output_path / "preview_latest.png");
        }
        
        // Force Refresh Button
        ImGui::SetCursorPos(ImVec2(10, 10)); // Top-left overlay
        if (ImGui::Button("Force Refresh")) {
             trainer.request_preview(pose, 200, 200, near_plane, far_plane);
        }

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    stop_training = true;
    training_thread.join();
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
