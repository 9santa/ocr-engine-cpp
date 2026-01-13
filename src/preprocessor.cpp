#include "preprocessor.h"
#include <algorithm>
#include <queue>

Preprocessor::Preprocessor() {}

// main processing function/pipeline
ImageMatrix Preprocessor::preprocess(const ImageMatrix& input) {
    ImageMatrix processed = applyGrayscale(input);
    processed = applyThreshold(processed);
    processed = removeNoise(processed);

    return processed;
}

// Grayscale to simplify processing, reduces data from 3 channels (RGB) to 1 Gray channel
ImageMatrix Preprocessor::applyGrayscale(const ImageMatrix& input) {
    if (input.channels == 1) return input;  // already Grayscale

    ImageMatrix gray(input.width, input.height, 1);
    if (input.channels == 3) {
        // Gray = 0.299*R + 0.587*G + 0.114*B
        for (int y = 0; y < input.height; y++) {
            for (int x = 0; x < input.width; x++) {
                float red = input(y, x, 0) / 255.0f;
                float green = input(y, x, 1) / 255.0f;
                float blue = input(y, x, 2) / 255.0f;
                float grayValue = 0.299f * red + 0.587f * green + 0.114f * blue;
                gray(y, x, 0) = static_cast<unsigned char>(grayValue * 255);
            }
        }

    }

    return gray;
}

// Binary Conversion.
// Converts grayscale image to binary (black&white) using 128 thresholding
ImageMatrix Preprocessor::applyThreshold(const ImageMatrix& input) {
    ImageMatrix binary(input.width, input.height, 1);

    const unsigned char threshold = 128;

    for (int y = 0; y < input.height; y++) {
        for (int x = 0; x < input.width; x++) {
            binary(y, x, 0) = (input(y, x, 0) > threshold) ? 255 : 0;
        }
    }

    return binary;
}

// Removes small noise particles using morphological operations
ImageMatrix Preprocessor::removeNoise(const ImageMatrix& input) {
    // first erode to remove small noise
    ImageMatrix eroded = morphologicalOperation(input, kernel, false);
    // then dilate to restore digit size
    ImageMatrix dilated = morphologicalOperation(eroded, kernel, true);


    return dilated;
}

// Morphological operation (erosion or dilation)
ImageMatrix Preprocessor::morphologicalOperation(const ImageMatrix& input, const std::vector<std::vector<int>>& kernel, bool isDilation) {
    ImageMatrix result(input.width, input.height, 1);
    int kernelHeight = (int)kernel.size();
    int kernelWidth = (int)kernel[0].size();
    int halfHeight = kernelHeight / 2;
    int halfWeight = kernelWidth / 2;

    for (int y = 0; y < input.height; y++) {
        for (int x = 0; x < input.width; x++) {
            unsigned char value = isDilation ? 0 : 255;

            for (int ky = 0; ky < kernelHeight; ky++) {
                for (int kx = 0; kx < kernelWidth; kx++) {
                    if (kernel[ky][kx] == 1) {
                        int ny = y + ky - halfHeight;
                        int nx = x + kx - halfWeight;

                        if (ny >= 0 && ny < input.height && nx >= 0 && nx < input.width) {
                            if (isDilation) {
                                // dilation -> take maximum value in kernel region
                                value = std::max(value, input(ny, nx, 0));
                            } else {
                                // erosion -> take minimum value in kernel region
                                value = std::min(value, input(ny, nx, 0));
                            }
                        }
                    }
                }
            }

            result(y, x, 0) = value;
        }
    }

    return result;
}


// Find digit contours using connected component analysys with BFS
std::vector<BoundingBox> Preprocessor::findDigitContours(const ImageMatrix& binary) {
    std::vector<std::vector<std::pair<int, int>>> components;
    findConnectedComponents(binary, components);

    std::vector<BoundingBox> digitBoxes;

    for (const auto& component : components) {
        if (component.empty()) continue;

        // find bounding box
        int minX = component[0].first, maxX = component[0].first;
        int minY = component[0].second, maxY = component[0].second;

        for (auto const& point : component) {
            minX = std::min(minX, point.first);
            maxX = std::max(maxX, point.first);
            minY = std::min(minY, point.second);
            maxY = std::max(maxY, point.second);
        }

        int width = maxX - minX + 1;
        int height = maxY - minY + 1;

        // filter by size to remove noise
        if (width >= minDigitWidth && height >= minDigitHeight && width <= maxDigitWidth && height <= maxDigitHeight) {
            digitBoxes.push_back({minX, minY, width, height});
        }
    }

    // sort boxes from left to right
    std::sort(digitBoxes.begin(), digitBoxes.end(), [](const BoundingBox& a, const BoundingBox& b) {
            return a.x < b.x;
            });

    return digitBoxes;
}


// Find connected components with BFS
void Preprocessor::findConnectedComponents(const ImageMatrix& binary, std::vector<std::vector<std::pair<int, int>>>& components) {
    std::vector<std::vector<bool>> visited(binary.height, std::vector<bool>(binary.width, false));

    // 8-connected neighborhood
    std::vector<std::pair<int, int>> directions = {
        {-1, -1}, {-1, 0}, {-1, 1},
        {0, -1},                {0, 1},
        {1, -1},  {1, 0},   {1, 1}
    };

    for (int y = 0; y < binary.height; y++) {
        for (int x = 0; x < binary.width; x++) {
            // if not visited and foreground pixel
            if (!visited[y][x] && binary(y, x, 0) > 128) {
                std::vector<std::pair<int, int>> component;
                std::queue<std::pair<int, int>> q;

                q.push({x, y});
                visited[y][x] = true;

                while (!q.empty()) {
                    auto current = q.front();
                    q.pop();
                    component.push_back(current);

                    for (const auto& dir : directions) {
                        int nx = current.first + dir.first;
                        int ny = current.second + dir.second;

                        if (nx >= 0 && nx < binary.width && ny >= 0 && ny < binary.height) {
                            if (!visited[ny][nx] && binary(ny, nx, 0) > 128) {
                                visited[ny][nx] = true;
                                q.push({nx, ny});
                            }
                        }
                    }
                }

                // minimum component size
                if ((int)component.size() > 10) {
                    components.push_back(component);
                }
            }
        }
    }
}


std::vector<ImageMatrix> Preprocessor::extractDigits(const ImageMatrix& image) {
    ImageMatrix processed = preprocess(image);
    std::vector<BoundingBox> digitBoxes = findDigitContours(processed);

    std::vector<ImageMatrix> digits;

    for (const auto& box : digitBoxes) {
        ImageMatrix digit(box.width, box.height, 1);

        for (int y = 0; y < box.height; y++) {
            for (int x = 0; x < box.width; x++) {
                int imageY = box.y + y;
                int imageX = box.x + x;
                if (imageY < processed.height && imageX < processed.width) {
                    digit(y, x, 0) = processed(imageY, imageX, 0);
                }
            }
        }

        // Resize to standard size for feature extraction
        ImageMatrix resized = resizeDigit(digit);

        digits.push_back(resized);
    }

    return digits;
}


ImageMatrix Preprocessor::resizeDigit(const ImageMatrix& digit, int targetWidth, int targetHeight) {
    ImageMatrix resized(targetWidth, targetHeight, 1);

    float scaleX = static_cast<float>(digit.width) / targetWidth;
    float scaleY = static_cast<float>(digit.height) / targetHeight;

    for (int y = 0; y < targetHeight; y++) {
        for (int x = 0; x < targetWidth; x++) {
            int srcX = static_cast<int>(x * scaleX);
            int srcY = static_cast<int>(y * scaleY);

            if (srcX < digit.width && srcY < digit.height) {
                resized(y, x, 0) = digit(srcY, srcX, 0);
            }
        }
    }

    return resized;
}


ImageMatrix Preprocessor::normalizeDigit(const ImageMatrix& digit) {
    //TODO
    return digit;
}
