#ifndef IMAGECOMPRESSION_H
#define IMAGECOMPRESSION_H

#include "RandomizedSVD.h"
#include <Eigen/Dense>
#include <iostream>

// from https://github.com/nothings/stb/tree/master
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
using namespace Eigen;

int main() {


  // Load the image using stb_image
  int width, height, channels;
  unsigned char* image_data = stbi_load(input_image_path, &width, &height,
                                        &channels, 3);  // Force load as RGB
  if (!image_data) {
    std::cerr << "Error: Could not load image " << input_image_path
              << std::endl;
    return 1;
  }

  std::cout << "Image loaded: " << width << "x" << height << " with "
            << channels << " channels." << std::endl;

  // Prepare Eigen matrices for each RGB channel
  MatrixXd red(height, width), green(height, width), blue(height, width);

}