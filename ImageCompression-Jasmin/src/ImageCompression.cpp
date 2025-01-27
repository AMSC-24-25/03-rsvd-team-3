#ifndef IMAGECOMPRESSIONG_H
#define IMAGECOMPRESSIONG_H

#include "RandomizedSVD.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>

// from https://github.com/nothings/stb/tree/master
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


using namespace Eigen;
using namespace std;

//Function to check if visually grayscale image has actually 3 channels
bool isGrayscaleRGB(unsigned char* image_data, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = (i * width + j) * 3;
            if (image_data[idx] != image_data[idx + 1] || image_data[idx] != image_data[idx + 2]) {
                return false;  // Not grayscale
            }
        }
    }
    return true;  // Visually grayscale
}

// Function to get the file size using the std::filesystem library
//long long getFileSize(const string& filename) {
//    return std::filesystem::file_size(filename);
//}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
    return 1;
  }

  const char* input_image_path = argv[1];
  int rank = std::stoi(argv[2]);  // Input rank


  // Load the image using stb_image
  int width, height, channels;
  unsigned char* image_data = stbi_load(input_image_path, &width, &height,
                                        &channels, 0); 
  if (!image_data) {
    cerr << "Error: Could not load image " << input_image_path << endl;
    return 1;
  }

  cout << "Image loaded: " << width << "x" << height << " with " << channels << " channels." << endl;

  bool visually_grayscale = (channels == 3) && isGrayscaleRGB(image_data, width, height);

  // Handle single-channel (grayscale) images directly
  vector<MatrixXd> channel_matrices(channels);
  if (channels == 1) {
      cout << "Input image is already grayscale with a single channel." << endl;
      channel_matrices[0].resize(height, width);
      for (int i = 0; i < height; ++i) {
          for (int j = 0; j < width; ++j) {
              channel_matrices[0](i, j) = static_cast<double>(image_data[i * width + j]) / 255.0;
          }
      }
  } else {
      // Check if the image is visually grayscale
      bool visually_grayscale = (channels == 3) && isGrayscaleRGB(image_data, width, height);

      if (visually_grayscale) {
          cout << "Image is visually grayscale, reducing to a single channel." << endl;
          channel_matrices.resize(1);
          channel_matrices[0].resize(height, width);
          for (int i = 0; i < height; ++i) {
              for (int j = 0; j < width; ++j) {
                  int idx = (i * width + j) * 3;
                  channel_matrices[0](i, j) = static_cast<double>(image_data[idx]) / 255.0;
              }
          }
          channels = 1;
      } else {
          cout << "Processing each channel separately." << endl;
          for (int c = 0; c < channels; ++c) {
              channel_matrices[c].resize(height, width);
              for (int i = 0; i < height; ++i) {
                  for (int j = 0; j < width; ++j) {
                      int idx = (i * width + j) * channels + c;
                      channel_matrices[c](i, j) = static_cast<double>(image_data[idx]) / 255.0;
                  }
              }
          }
      }
 }

  //Computing input image size (indipendently of image format)
  double input_size = width * height * channels * sizeof(unsigned char) / 1024.0; // KB
  
  // Free the original image data
  stbi_image_free(image_data);

  vector<MatrixXd> compressed_channels(channels);

  // Compress each channel
  for (int c = 0; c < channels; ++c) {
    cout << "Compressing channel " << c + 1 << "..." << endl;

    // Create an instance of RandomizedSVD
    Eigen::RandomizedSVD<MatrixXd> rSVD;

    // Compute the SVD for the current channel matrix
    rSVD.compute(channel_matrices[c], rank);

    // Reconstruct the compressed channel
    compressed_channels[c] = rSVD.matrixU() * rSVD.singularValues().asDiagonal() * rSVD.matrixV().transpose();
}

  // Combine and save the compressed image
  std::vector<unsigned char> compressed_image_data(width * height * channels);
  for (int c = 0; c < channels; ++c) {
      for (int i = 0; i < height; ++i) {
          for (int j = 0; j < width; ++j) {
              int idx = (i * width + j) * channels + c;
              compressed_image_data[idx] = static_cast<unsigned char>(
                  min(255.0, max(0.0, compressed_channels[c](i, j) * 255.0)));
          }
      }
  }

    // Save the compressed image
    const std::string output_image_path = "../test_image/compressed_image.png";
    if (stbi_write_png(output_image_path.c_str(), width, height, channels, compressed_image_data.data(), width * channels) == 0) {
        std::cerr << "Error: Could not save the compressed image" << std::endl;
        return 1;
    }

 cout << "Image successfully saved to " << output_image_path << endl;
 // Memory usage calculation for the compressed matrices (U, Sigma, V)
 size_t element_size = sizeof(double);  // Each element is a double (8 bytes)
 size_t size_U = height * rank * element_size;
 size_t size_Sigma = rank * rank * element_size;
 size_t size_V = rank * width * element_size;
 size_t total_size = (size_U + size_Sigma + size_V) * channels;

 // Convert to MB
 double total_size_KB = total_size / (1024.0);
 double compression_ratio = input_size/total_size_KB;
 cout << "Original input size: " << input_size << "KB" << endl;
 cout << "Memory usage of compressed matrices: " << total_size_KB << "KB" << endl;
 cout << "Memory compression ratio: "<< compression_ratio << endl;
 cout << "Memory saving: " << input_size-total_size_KB << "KB" << endl;


 // Compute the relative Frobenius norm error
 double total_relative_error = 0.0;

 for (int c = 0; c < channels; ++c) {
     // Compute Frobenius norm of the original channel
     double original_norm = std::sqrt(channel_matrices[c].squaredNorm());

    // Compute Frobenius norm of the difference
     double difference_norm = std::sqrt((channel_matrices[c] - compressed_channels[c]).squaredNorm());

     // Compute relative error for the channel
     double relative_error = difference_norm / original_norm;
    total_relative_error += relative_error;
 }

  // Compute average relative error across all channels, so that we can compare errors between grayscale and RGB images
 double average_relative_error = total_relative_error / channels;

 cout << "Average relative Frobenius norm error: " << average_relative_error << endl;

 return 0;

}

#endif