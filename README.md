# Image Segmentation using DFT Features
## Overview

This Python script implements a block-based image segmentation and classification system for fruits using **Discrete Fourier Transform (DFT)** features. It offers two primary functionalities:

1.  **DFT Feature-based Segmentation:** Classifies image blocks by comparing their DFT features to pre-trained mean feature vectors of known fruit classes using Euclidean distance. This approach includes dynamic thresholding to identify background regions.
2.  **Simple Decision Tree Classification:** Utilizes extracted DFT features to train and evaluate a Simple Decision Tree Classifier for multi-class fruit classification.

The script is designed to handle varying numbers of fruit classes (demonstrated with 3, 5, and 10 fruits) and divides the image into 16x16 block sizes for feature extraction. Here in this script 16X16 is used because of its computational efficiency and better representation of global features and textures.

---

## Features

* **DFT Feature Extraction:** Calculates four distinct features from the 2D DFT magnitude spectrum (DC component, sum of low-frequency coefficients, average mid-frequency magnitude, and standard deviation of magnitudes) for each R, G, B channel of an image block, concatenating them into a single feature vector.
* **Mean Feature Vector Training:** Computes mean DFT feature vectors for each fruit class from a training dataset, forming the basis for the DFT-based segmentation.
* **Euclidean Distance Classification:** Classifies individual image blocks by finding the fruit class whose mean feature vector has the minimum Euclidean distance to the block's feature vector.
* **Dynamic Background Thresholding:** Incorporates a dynamic threshold during segmentation to classify blocks as "Background" if their minimum distance to any fruit class exceeds a defined value, improving distinction from non-fruit regions.
* **Block-Based Segmentation:** Divides input collage images into uniform blocks and classifies each block to generate a segmentation map.
* **Segmentation Visualization & Saving:** Generates visual comparisons of original and segmented images using `matplotlib` and saves the segmented output to a specified directory.
* **Simple Decision Tree Classification:** Implements a Simple Decision Tree Classifier using `scikit-learn` to perform supervised classification of fruit images based on DFT features extracted from their blocks.
* **Model Evaluation:** Provides comprehensive evaluation for the Decision Tree model, including a classification report (precision, recall, F1-score) and a visually enhanced confusion matrix using `seaborn`, which is also saved. 
* **Configurable Paths:** Allows easy customization of paths for training data, test collages, and output directories.
* **Scalability:** Demonstrates functionality with sets of 3, 5, and 10 fruit classes.

---

## Requirements

* Python 3.x
* OpenCV
* NumPy
* Matplotlib (for creating plots)
* SciPy (for DFT)
* Scikit-learn (Splitting Data into training and testing sets in Decision Tree implementation and for evaluation of model perfromance)
* Seaborn (For enhanced visualization of the confusion matrix)

---

## Usage

1.  **Prepare the data:**
    * Use the unzipped  `Train Data.zip` directory added in this repo or Create a `Train Data` directory containing subfolders for each fruit class (e.g., `Train Data/Apple Red 1/`, `Train Data/Mango/`), with relevant training images inside.
    * Use the  `TestData` directory added in this repo or Create a `TestData` directory containing the collage images, as specified by the `TEST_COLLAGE_PATH_X` variables.
    * Ensure all image files are in supported formats (`.jpg`, `.png`, `.jpeg`).

2.  **Run the script:**
    * Open the terminal or command prompt.
    * Navigate to the directory where the script is located.
    * Execute the script

3.  **View the results:**
    * The script will create a `Result_Dir` folder (as specified by `RESULTS_DIR`) in the same directory.
    * This folder will contain:
        * Segmented versions of test collage images (e.g., `3_Fruits_Segmentation_Block_Size_16x16_segmented.jpg`).
        * PNG images of confusion matrices for the Decision Tree classification (e.g., `3_fruits_confusion_matrix.png`).
    * The console will display training progress and mean feature vectors for Euclidean distance based Classifier and classification reports for the Decision Tree model.

---

## Script Details

### `extract_dft_features(block)`

* **Purpose:** Extracts DFT-based features from a 3-channel image block.
* **Process:** Computes 2D DFT for each channel, shifts the zero-frequency component, calculates the magnitude spectrum, and then extracts DC component, sum of low-frequency coefficients, average mid-frequency magnitude, and standard deviation of non-DC magnitudes. Features from all channels are concatenated.

### `train_dft_classifier(fruits_list, train_base_path, block_size)`

* **Purpose:** Trains the DFT-based classifier by calculating the mean feature vector for each fruit class.
* **Process:** Iterates through specified fruit folders, loads images, divides them into blocks, extracts features from each block, and computes the average feature vector for each fruit across all its training blocks.

### `classify_block(block_features, fruit_features_means, threshold=100000)`

* **Purpose:** Classifies a single image block based on its DFT features using Euclidean distance.
* **Process:** Compares the input `block_features` to the `mean_features` of each trained fruit, assigning the block to the class with the minimum Euclidean distance. Blocks are classified as "Background" if the minimum distance exceeds a `threshold`.

### `perform_block_segmentation(img_path, trained_means, fruits_list, block_size)`

* **Purpose:** Performs block-based image segmentation on a collage image.
* **Process:** Loads the image, converts it to RGB, divides it into blocks, extracts features for each block, and uses `classify_block` to determine the class for each block. It creates a `segmentation_map` (an array of class indices) and adjusts the background `threshold` dynamically based on `block_size`.

### `visualize_block_segmentation(segmentation_map, original_img_rgb, fruits_list, block_size=16, title="Segmentation Result")`

* **Purpose:** Visualizes the generated block-based segmentation map alongside the original image.
* **Process:** Creates a color-coded image from the `segmentation_map` using predefined colors for each fruit class and black for the background. Displays both images using `matplotlib.pyplot` and saves the segmented image to a file.

### `load_images_for_class(class_name)`

* **Purpose:** Loads and preprocesses images for a specific fruit class for Decision Tree training.
* **Process:** Reads images from the designated folder, resizes them to 100x100 pixels, and returns lists of image arrays and their corresponding class labels.

### `divide_into_blocks(img, block_size=16)`

* **Purpose:** Divides an input image into non-overlapping square blocks for Decision Tree Classifier.
* **Process:** Iterates through the image dimensions, extracting `block_size` x `block_size` portions and appending them to a list.

### `evaluate_model(y_test, y_pred, title)`

* **Purpose:** Evaluates and displays the performance of the Decision Tree Classifier.
* **Process:** Prints a `classification_report` (including precision, recall, f1-score) and generates, and saves a `confusion_matrix` using `seaborn` for enhanced visualization.

### `SimpleDecisionTree(BLOCK_SIZE=16)`

* **Purpose:** Orchestrates the training and evaluation of the Decision Tree Classifier for different sets of fruits.
* **Process:** Iterates through 3, 5, and 10 fruit sets. For each set, it loads images, divides them into blocks, extracts DFT features, splits data into training/testing sets, trains a `DecisionTreeClassifier`, makes predictions, and calls `evaluate_model` to report performance.

### `main(BLOCK_SIZE=16)`

* **Purpose:** The main execution function for the DFT-based segmentation process.
* **Process:** Runs in three phases (3, 5, and 10 fruits). For each phase, it sets the current fruit list, trains the DFT classifier, performs block-based segmentation on the corresponding test collage, and saves the results.

---

## Notes

* The `threshold` used in `classify_block` for background separation is dynamically set within `perform_block_segmentation` based on the chosen `BLOCK_SIZE`. It can be adjusted if fine-tuning is required.
* The Decision Tree classifier's `max_depth` is set to 10 by default in `SimpleDecisionTree`. This can be adjusted to control model complexity and prevent overfitting/underfitting.
* The `BLOCK_SIZE` parameter in the `main()` and `SimpleDecisionTree()` calls at the end of the script can be modified to experiment with different block dimensions (e.g., `BLOCK_SIZE=8` or `BLOCK_SIZE=4`).

---
