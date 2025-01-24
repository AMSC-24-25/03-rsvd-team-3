#ifndef IMAGECOMPRESSIONG_H
#define IMAGECOMPRESSIONG_H

#include "RandomizedSVD.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

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
    if (stbi_write_png(output_image_path.c_str(), width, height, channels, compressed_image_data, width * channels) == 0) {
        std::cerr << "Error: Could not save the compressed image" << std::endl;
        delete[] compressed_image_data;
        return 1;
    }

 cout << "Image successfully saved to " << output_image_path << endl;

 delete[] compressed_image_data;
 return 0;

}

#endif