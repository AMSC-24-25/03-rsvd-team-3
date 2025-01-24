# Image Compression
Implementation of Image Compression using the RandomizedSVD algorithm with a GivensRotationQR. The basic idea is to approximate the image matrix with a low-rank matrix using RSVD and store only the significant singular values and vectors.

Main Steps:
1. Load the image as a 2D matrix (for grayscale with 1 channel) or a 3D tensor (for RGB). For RGB, check if all three channels are equal, if so, the image is actually visually grayscale. Thus, only keep one channel and lighten computation.
2. Preprocessing: normalization of values to (0,1) interval and, if the image si RGB, splitting into three separate matrices.
3. Apply rSVD
4. Compress the data: store the truncated images instead of the full matrix, reducing the memory space required for storing.
5. Postprocessing: rescale the pixel values back to their original range (0, 255) and, for RGB images, combine the three channels again.
6. Save and Display the compressed image.

As input to the function we give the image path and the rank we want to use for the application of RandomizedSVD. It determines the number of singular values retained, so, it directly controls the level of compression, thus, the quality of the compressed image. In general: 
- when the rank is about 5-15% of the image minimum size, we get a visually obvious compression with often introduced artifacts.
- when the rank is around 15%-20% of the image minimum size, the image maintains a medium-level quality and compression requires some attention to be spot.
- when the rank is above 20% of the image minimum size, the compressed output maintains a high quality and is very difficult to spot.