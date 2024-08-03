# Neural Network Watermarking Tool

This repository contains Python scripts for embedding and extracting watermarks in the Least Significant Bits (LSB) of weights in Fully Connected Neural Networks (FCNN) or Convolutional Neural Networks (CNN) such as ResNet.

## Features

- Embed various types of watermarks (images, text) into neural network models
- Flexible embedding patterns and positioning
- Secure watermark verification using key pairs
- Watermark extraction and integrity checking

## Scripts

### 1. Watermark Embedding (`watermark_inserter_LSB.py`)

This script allows you to insert a watermark into the LSB of weights in a neural network model.

#### Input
- Neural network model
- Watermark (image or text)

#### Customization Options
- **Embedding Pattern**: Choose to embed in even or odd indices, or sequentially without skips
- **Padding**: Specify the starting position of the watermark within the model
- **Security Keys**: Two keys randomly generated are inserted (before and after the watermark) to ensure integrity

#### Output
- Watermarked model
- Key file for watermark retrieval

### 2. Watermark Extraction (`watermark_checker_LSB.py`)

This script retrieves the embedded watermark from a neural network model.

#### Input
- Watermarked neural network model
- Key file (generated during embedding)

#### Output
- Extracted watermark
- Integrity verification result

## Performance

While not fully optimized, these scripts offer reasonable performance. Users are encouraged to modify and improve the code as needed, with proper attribution.

## Recommendations

Always visually inspect the extracted watermark, especially for image watermarks, as human perception can often detect subtle alterations that may not be caught by automated integrity checks.

## Contribution

Feel free to fork this repository, make improvements, and submit pull requests. We appreciate any contributions that enhance the functionality, performance, or usability of these scripts.

## Citation

If you use or modify this code for your projects or research, please include a citation to this repository.

