import os
import cv2
import numpy as np
import matplotlib.pyplot as plt #For Plotting and visualization
from scipy.fft import fft2, ifft2 #For Clarity and Flexibilty, using Scipy's FFT
from sklearn.tree import DecisionTreeClassifier #For simple decision tree implementation
from sklearn.model_selection import train_test_split #Splitting Data into training and testing sets(For Decision Tree implementation only)
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay #For evaluation of model perfromance(Decision Tree)
import seaborn as sns #For enhanced visualization of the confusion matrix

# Configurable Paths
TRAIN_BASE = r".\TrainData" #Training Images path
TEST_COLLAGE_PATH_3= r".\TestData\NewCollage3Fruits.jpg"# Collage with 3 fruits
TEST_COLLAGE_PATH_5= r".\TestData\NewCollage5Fruits.jpg" # Collage with 5 fruits
TEST_COLLAGE_PATH_10= r".\TestData\NewCollage10Fruits.jpg" # Collage with 10 fruits
RESULTS_DIR = r".\Result_Dir" #Folder to store results
os.makedirs(RESULTS_DIR, exist_ok=True) #For creating a directory, if doesn't already exists

# Selected fruits for demonstration (starting with 3, then expanding to 5 and 10)
FRUITS_3 = ['Apple Red 1', 'Mango', 'Blueberry']
FRUITS_5 = ['Apple Red 1', 'Mango', 'Blueberry', 'Lemon', 'Kiwi']
FRUITS_10 = ['Apple Red 1', 'Mango', 'Blueberry', 'Lemon', 'Kiwi', 'Orange', 'Peach', 'Plum', 'Pineapple', 'Tomato 1']

# Global variable to hold the currently active fruit list
CURRENT_FRUITS = [] #Variable to keep track of processed Fruit List

# Function to extract DFT features from a block (3 channels-R,G,B)
def extract_dft_features(block):
    """
    Extracts features from the 2D Discrete Fourier Transform (DFT) of an image block.
    This version processes each channel (R, G, B) and concatenates their features.

    Args:
        block (np.array): A 3-channel (e.g., BGR or RGB) image block.

    Returns:
        np.array: A 1D array of selected DFT features concatenated for all channels.
    """
    all_channel_features = [] #Empty List to store features

    # Process each channel independently
    for i in range(block.shape[2]): # Iterate through channels (0, 1, 2 for R,G,B)
        channel = block[:, :, i] #To extract single channel
        block_float = np.float32(channel) #Convertion of channel data to float for FFT computation

        # Compute 2D DFT
        dft = fft2(block_float) #2D Fast Fourier Transformation
        dft_shifted = np.fft.fftshift(dft) #Shift the zero-frequency component to center of the spectrum
        magnitude_spectrum = np.abs(dft_shifted) #Magnitude Spectrum extraction from shifted DFT

        center_y, center_x = channel.shape[0] // 2, channel.shape[1] // 2 #Determining centre coordinates of the spectrum

        # Feature 1: DC component (average intensity)
        dc_component = magnitude_spectrum[center_y, center_x]

        # Feature 2: Sum of magnitudes of a few low-frequency coefficients (e.g., 4 neighbors of DC)
        sum_low_freq = 0.0 #Initialization
        if channel.shape[0] > 1 and channel.shape[1] > 1: #Ensure the Block is large for the neighbors
            if center_x + 1 < channel.shape[1]:
                sum_low_freq += magnitude_spectrum[center_y, center_x + 1]
            if center_x - 1 >= 0:
                sum_low_freq += magnitude_spectrum[center_y, center_x - 1]
            if center_y + 1 < channel.shape[0]:
                sum_low_freq += magnitude_spectrum[center_y + 1, center_x]
            if center_y - 1 >= 0:
                sum_low_freq += magnitude_spectrum[center_y - 1, center_x]
        
        # Feature 3: Average magnitude of selected mid-frequency coefficients
        mid_freq_mag = 0.0 #Initialization
        k = 2 # Example offset for mid-range frequencies
        coords_to_check = [
            (center_y + k, center_x + k), (center_y + k, center_x - k),
            (center_y - k, center_x + k), (center_y - k, center_x - k)
        ]
        count_mid_freq = 0
        for r, c in coords_to_check:
            if 0 <= r < channel.shape[0] and 0 <= c < channel.shape[1]:
                mid_freq_mag += magnitude_spectrum[r, c]
                count_mid_freq += 1
        if count_mid_freq > 0:
            mid_freq_mag /= count_mid_freq
        
        # Feature 4: Standard deviation of magnitudes across the entire spectrum (excluding DC)
        flat_mag_spectrum = magnitude_spectrum.flatten() #Flatten magnitude spectrum into 1D array
        dc_index = center_y * channel.shape[1] + center_x #1D index of DC component
        non_dc_magnitudes = np.delete(flat_mag_spectrum, dc_index)
        std_dev_mag = np.std(non_dc_magnitudes) if len(non_dc_magnitudes) > 0 else 0.0 #Extract Standard Deviation, handling empty array case

        # Combine features for this channel
        channel_features = [
            dc_component,
            sum_low_freq,
            mid_freq_mag,
            std_dev_mag
        ]
        all_channel_features.extend(channel_features)
    
    return np.array(all_channel_features) #Return Features as a numpy array

# Function for Training Fruits according to the DFT coefficients, block by block 
def train_dft_classifier(fruits_list, train_base_path, block_size): 
    """
    Trains the DFT-based classifier by calculating the mean feature vector for each fruit.

    Args:
        fruits_list (list): List of fruit names to train for.
        train_base_path (str): Path to the training dataset.
        block_size (int): Size of the square blocks for DFT.

    Returns:
        dict: A dictionary where keys are fruit names and values are their mean feature vectors.
    """
    fruit_features_means = {} #Dictionary for storing mean feature vector for each fruit

    print(f"Training classifier for {len(fruits_list)} fruits with block size {block_size}x{block_size}...")

    for fruit in fruits_list: #Iterating through each fruit in the list
        folder = os.path.join(train_base_path, fruit)
        if not os.path.isdir(folder): #if folder isn't found
            print(f"Warning: Training folder for {fruit} not found at {folder}. Skipping.")
            continue

        all_fruit_block_features = [] #List for storing all the features from each block
        files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        num_images_to_process = len(files) #Total Number of Images for current Fruit
        print(f"  Processing {num_images_to_process} images for {fruit}...")

        for i, file in enumerate(files[:num_images_to_process]):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path) 
            if img is None: #Check if the image is found or not
                print(f"Could not read image: {img_path}")
                continue
            
            # Divide the image into blocks and extract features
            h, w, _ = img.shape # Now img has 3 channels
            for r in range(0, h - block_size + 1, block_size):
                for c in range(0, w - block_size + 1, block_size):
                    # Extract 3-channel block
                    block = img[r:r + block_size, c:c + block_size, :] 
                    
                    if block.shape[0] == block_size and block.shape[1] == block_size:
                        features = extract_dft_features(block) # Pass 3-channel block
                        all_fruit_block_features.append(features) #Add the features into the list

        if all_fruit_block_features: 
            fruit_features_means[fruit] = np.mean(all_fruit_block_features, axis=0) #calculates the mean feature vector for each fruit
            print(f"  Calculated mean features for {fruit}: {fruit_features_means[fruit]}")
        else:
            print(f"  No features extracted for {fruit}.")

    return fruit_features_means #Returns the dictionary containing mean feature vectors

# Based on Euclidean Distance, doing classification between different Fruit classes
def classify_block(block_features, fruit_features_means,threshold=100000):
    """
    Classifies a single block based on its DFT features using Euclidean distance.

    Args:
        block_features (np.array): Features of the current block.
        fruit_features_means (dict): Dictionary of mean feature vectors for each fruit.
        threshold(float): the Boundary condition of each fruit mean vector(if it is bigger than threshold then it is predicted as 'Background') 

    Returns:
        str: The predicted fruit class name, or "Background" if no close match.
    """
    min_distance = float('inf') #Initialization of distance to infinity
    predicted_class = "Background" # Default for unclassified blocks

    if not fruit_features_means: #If no fruits means, then it is the background
        return predicted_class

    for fruit, mean_features in fruit_features_means.items():
        if len(block_features) != len(mean_features):
            print(f"Dimension mismatch: Block features {len(block_features)}, Mean features {len(mean_features)}. Skipping classification for this block.")
            continue
            
        distance = np.linalg.norm(block_features - mean_features) #Calculation of Euclidean Distance between block features and mean features
            
        if distance < min_distance: 
            min_distance = distance
            predicted_class = fruit #Update Predicted Class
        if min_distance > threshold: 
            predicted_class = "Background" #Classifying as "Background" for distance greater than threshold
            
    return predicted_class #Return the class name
 
# In the image collage, performs block by block segmentation and returns segmented image 
def perform_block_segmentation(img_path, trained_means, fruits_list, block_size):
    """
    Performs block-based segmentation on an input image(collage).

    Args:
        img_path (str): Path to the image to segment.
        trained_means (dict): Dictionary of mean feature vectors for each fruit.
        fruits_list (list): List of fruit names in the order of training.
        block_size (int): Size of the square blocks.

    Returns:
        tuple: (segmentation_map, original_img_rgb, original_img_bgr)
        segmentation_map (np.array): An array where each element represents the class index of the corresponding block. -1 for background.
        original_img_rgb (np.array): The loaded original image in RGB format.
        original_img_bgr (np.array): The loaded original image in BGR format (for saving).
    """
    original_img_bgr = cv2.imread(img_path)
    if original_img_bgr is None: #If there is no image found
        print(f"Error: Could not load image from {img_path}")
        return None, None, None

    img_rgb = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB)
    
    h, w, _ = original_img_bgr.shape # Get dimensions including channels
    
    num_blocks_h = h // block_size #Number of blocks vertically
    num_blocks_w = w // block_size #Number of blocks horizontally
    
    segmentation_map = np.full((num_blocks_h, num_blocks_w), -1, dtype=int) #Initialize with -1
    
    fruit_to_idx = {fruit: i for i, fruit in enumerate(fruits_list)} #Mapping from fruit name to integer index
    
    print(f"Segmenting image {img_path} into {num_blocks_h}x{num_blocks_w} blocks...")
    
    for r_idx in range(num_blocks_h):
        for c_idx in range(num_blocks_w):
            r_start = r_idx * block_size #Starting with row pixel coordinates
            c_start = c_idx * block_size #Starting with column pixel coordinates 
            # Extract 3-channel block
            block = original_img_bgr[r_start : r_start + block_size, c_start : c_start + block_size, :] #Extraction of the current block
            
            #Dynamic threshold adjustment based on block size
            if(block_size==16):
                threshold=40000 #Threshold for 16X16 blcoks
            elif(block_size==8):
                threshold=20000 #Threshold for 16X16 blcoks
            elif(block_size==4):
                threshold=5000 #Threshold for 16X16 blcoks
            
            if block.shape[0] == block_size and block.shape[1] == block_size: #Ensuring the block is of full size
                features = extract_dft_features(block) # Pass 3-channel block
                predicted_class_name = classify_block(features, trained_means, threshold) #Classifying the block
                
                if predicted_class_name in fruit_to_idx:
                    segmentation_map[r_idx, c_idx] = fruit_to_idx[predicted_class_name] #Assigning it's corresponding index
                else:
                    segmentation_map[r_idx, c_idx] = -1 # Explicitly set to -1 for background
    
    return segmentation_map, img_rgb, original_img_bgr #Return the map along with images 

# The visualize_block_segmentation function remains largely the same as it already expects RGB for display.
def visualize_block_segmentation(segmentation_map, original_img_rgb, fruits_list, block_size=16,title="Segmentation Result"):
    """
    Visualizes the block-based segmentation map.
    
    Args:
        segmentation_map (np.array): The array representing the class index of each block.
        original_img_rgb (np.array): The original image (in RGB format for matplotlib).
        fruits_list (list): List of fruit names corresponding to class indices.
        block_size(int): Size of the square blocks.
        title (str): Title for the plot.
    """
    colors = {
        -1: (0, 0, 0),       # Background: Black
        0: (255, 0, 0),      # Apple Red 1: Red
        1: (0, 100, 0),      # Mango: Dark Green 
        2: (0, 0, 255),      # Blueberry: Blue 
        3: (255, 255, 0),    # Lemon: Yellow 
        4: (139, 69, 19),    # Kiwi: Brown 
        5: (255, 165, 0),    # Orange: Orange 
        6: (255, 192, 203),  # Peach: Rose 
        7: (238, 130, 238),  # Plum: Violet 
        8: (144, 238, 144),  # Pineapple: Light Green 
        9: (255, 99, 71)     # Tomato 1: Light Red 
      }

    h_blocks, w_blocks = segmentation_map.shape #Dimentions of the segmentation map

    segmented_color_img = np.zeros(original_img_rgb.shape, dtype=np.uint8) #Initiating with a black image of same size

    for r_idx in range(h_blocks): #Iteration through rows
        for c_idx in range(w_blocks): #Iteration through columns
            class_idx = segmentation_map[r_idx, c_idx] #Class index for current block
            color = colors.get(class_idx, (0, 0, 0)) # Default to black if class_idx not found

            r_start = r_idx * block_size #Calculate starting row pixel coordinate
            c_start = c_idx * block_size #Calculate starting column pixel coordinate
            
            # Ensure indices don't go out of bounds for the original image size
            r_end = min(r_start + block_size, original_img_rgb.shape[0])
            c_end = min(c_start + block_size, original_img_rgb.shape[1])
            
            # Fill the corresponding block in the output image with the class color
            segmented_color_img[r_start : r_end, c_start : c_end] = color

    #Visualization of Original and Segmented Image
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_img_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(title)
    plt.imshow(segmented_color_img)
    plt.axis('off')
    save_path = os.path.join(RESULTS_DIR, f"{title.replace(' ', '_')}_Original_and_Segmented_Image.png")
    plt.savefig(save_path, bbox_inches='tight') #Save the Original_and_Segmented_Image
    plt.close()
    
    # Save the segmented image
    segmented_bgr = cv2.cvtColor(segmented_color_img, cv2.COLOR_RGB2BGR)
    #Saves segmented image into the folder based on specific file name
    cv2.imwrite(os.path.join(RESULTS_DIR, f"{title.replace(' ', '_').replace(':', '')}_segmented.jpg"), segmented_bgr) 


#The below functions are only for the implementation of Decision Tree Classifier 
# Function to load images from a given fruit class
def load_images_for_class(class_name):
    """
    Loads all images for a specified fruit class from the training base path.
    Resizes images to 100x100 pixels.

    Args:
        class_name (str): The name of the fruit class (e.g., 'Apple Red 1').

    Returns:
        tuple: A tuple containing two lists:
            - images (list): A list of loaded image NumPy arrays.
            - labels (list): A list of class names, one for each image.
    """
    folder = os.path.join(TRAIN_BASE, class_name) 
    images, labels = [], [] #Initializing Images and Labels list
    for i, file in enumerate(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, file)) 
        img = cv2.resize(img, (100, 100))
        images.append(img) #Adding the image to the list images
        labels.append(class_name) #Adding the label to the list labels 
    return images, labels #Returning the list of images and labels 

# Divide image into square blocks 
def divide_into_blocks(img, block_size=16):
    """
    Divides an input image into non-overlapping square blocks of a specified size.

    Args:
        img (np.array): The input image array.
        block_size (int): The desired size of the square blocks.

    Returns:
        list: A list of image blocks, where each block is a NumPy array.
    """
    h, w, _= img.shape #Image dimentions
    blocks = [] #Initializing blocks list
    for i in range(0, h, block_size): #Iterate through rows
        for j in range(0, w, block_size): #Iterate through columns 
            block = img[i:i+block_size, j:j+block_size] #Extraction of current block
            if block.shape[0] == block_size and block.shape[1] == block_size: #Ensuring the size of the block
                blocks.append(block)
    return blocks #Return the list of blocks

#Function to evaluate the model performance, display and save the results
def evaluate_model(y_test, y_pred, title):
    """
    Prints a classification report, displays a confusion matrix, and saves the confusion matrix.

    Args:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted labels.
        title (str): Title for the classification report and confusion matrix plot.
    """
    print(f"\n--- {title} Classification Report ---") 
    print(classification_report(y_test, y_pred, zero_division=0)) #Printing classification report handling zero division
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test)) #Compute Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test)) #Display object for confusion matrix
    disp.plot(cmap=plt.cm.Blues) #Plot Confusion Matrix
    plt.title(title + " - Confusion Matrix")
    save_path = os.path.join(RESULTS_DIR, f"{title.replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(save_path, bbox_inches='tight') #Save the confusion matrix plot 

def SimpleDecisionTree(BLOCK_SIZE=16):
    """
    Trains and evaluates a Decision Tree Classifier for different sets of fruits
    (3, 5, and 10 fruits) using DFT features extracted from image blocks.
    """
    # Test sets
    TEST_SETS = {
            "3 fruits": ['Apple Red 1', 'Mango', 'Blueberry'],
            "5 fruits": ['Apple Red 1', 'Mango', 'Blueberry', 'Lemon', 'Kiwi'],
            "10 fruits": ['Apple Red 1', 'Mango', 'Blueberry', 'Lemon', 'Kiwi', 'Orange', 'Peach', 'Plum', 'Pineapple', 'Tomato 1']
        }
    
    # Loop through each set for 3, 5, 10 fruits
    for set_name, fruits in TEST_SETS.items():
        print(f"\n==============================")
        print(f"Training on: {set_name}")
        print(f"==============================")
    
        block_size=BLOCK_SIZE #Set block size for this iteration
        
        all_images, all_labels = [], [] #List for images and their corresponding labels
        for fruit in fruits:
            imgs, lbls = load_images_for_class(fruit) #Loading fruits and labels
            all_images.extend(imgs)
            all_labels.extend(lbls)
    
        X, y = [], [] #Initialize lists to store all features and labels for block data
        for img, label in zip(all_images, all_labels): 
            blocks = divide_into_blocks(img, block_size) #Division of images into blocks
            for block in blocks:
                y.append(label) #Append the image labels into the set 'y'
                
        for img in all_images:
            h, w, _= img.shape #Get image dimentions
            blocks = []
            for i in range(0, h, block_size):
                for j in range(0, w, block_size):
                    block = img[i:i+block_size, j:j+block_size]
                    if block.shape[0] == block_size and block.shape[1] == block_size:
                        features=extract_dft_features(block) #DFT feature extraction
                        X.append(features)
        y = np.array(y) #Making it into a numpy array
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #Spliting as 80% for training and 20% for testing
    
        clf = DecisionTreeClassifier(max_depth=10, random_state=42) #Initialize decision tree classifier with depth 10
        clf.fit(X_train, y_train) #Training classifier with features and labels
        y_pred = clf.predict(X_test) #Making predictions on test set
    
        evaluate_model(y_test, y_pred, set_name) #Evaluate and display model's performance 

def main(BLOCK_SIZE=16): 
    """
    Performs the entire image segmentation process using DFT features.

    This function performs the following steps in three phases (3, 5, and 10 fruits):
    1. Sets the list of fruits to be considered for the current phase.
    2. Trains a DFT-based classifier by calculating mean feature vectors for each fruit
       using training images divided into blocks.
    3. If training is successful, performs block-based segmentation on a corresponding
       test collage image.
    4. Visualizes the segmentation results (original image vs. segmented map) and
       saves the segmented image.

    Args:
        BLOCK_SIZE (int): The size of the square blocks (e.g., 16 for 16x16 pixels) into which images are divided for feature extraction and segmentation.
    """
    # --- Phase 1: 3 Fruits ---
    print("\n--- Running Segmentation for 3 Fruits ---")
    CURRENT_FRUITS = FRUITS_3 #Assigned to list of 3 fruits
    
    trained_fruit_means_3 = train_dft_classifier(CURRENT_FRUITS, TRAIN_BASE, BLOCK_SIZE) #Training the classifier with 3 Fruits
    
    if trained_fruit_means_3:
        #Performing Segmentation on 3 fruit collage
        segmentation_map_3, original_img_rgb_3, original_img_bgr_3 = perform_block_segmentation(TEST_COLLAGE_PATH_3, trained_fruit_means_3, CURRENT_FRUITS, BLOCK_SIZE) 
        if segmentation_map_3 is not None:
            #Visualize and Save the 3 Fruit Segmentation
            visualize_block_segmentation(segmentation_map_3, original_img_rgb_3, CURRENT_FRUITS,BLOCK_SIZE, title=f"3 Fruits Segmentation (Block Size {BLOCK_SIZE}x{BLOCK_SIZE})")
        else:
            print(f"Skipping 3-fruit visualization due to segmentation error.")
    else:
        print("Skipping 3-fruit segmentation due to no trained models.")

    # --- Phase 2: 5 Fruits ---
    print("\n--- Running Segmentation for 5 Fruits ---")
    CURRENT_FRUITS = FRUITS_5 #Assigned to list of 5 fruits
    
    trained_fruit_means_5 = train_dft_classifier(CURRENT_FRUITS, TRAIN_BASE, BLOCK_SIZE) #Training the classifier with 3 Fruits
    
    if trained_fruit_means_5:
        #Performing Segmentation on 5 fruit collage
        segmentation_map_5, original_img_rgb_5, original_img_bgr_5 = perform_block_segmentation(TEST_COLLAGE_PATH_5, trained_fruit_means_5, CURRENT_FRUITS, BLOCK_SIZE)
        if segmentation_map_5 is not None:
            #Visualize and Save the 5 Fruit Segmentation
            visualize_block_segmentation(segmentation_map_5, original_img_rgb_5, CURRENT_FRUITS, BLOCK_SIZE, title=f"5 Fruits Segmentation (Block Size {BLOCK_SIZE}x{BLOCK_SIZE})")
        else:
            print(f"Skipping 5-fruit visualization due to segmentation error.")
    else:
        print("Skipping 5-fruit segmentation due to no trained models.")

    # --- Phase 3: 10 Fruits ---
    print("\n--- Running Segmentation for 10 Fruits ---")
    CURRENT_FRUITS = FRUITS_10 #Assigned to list of 10 fruits
    
    trained_fruit_means_10 = train_dft_classifier(CURRENT_FRUITS, TRAIN_BASE, BLOCK_SIZE) #Training the classifier with 10 Fruits
    
    if trained_fruit_means_10:
        #Performing Segmentation on 10 fruit collage
        segmentation_map_10, original_img_rgb_10, original_img_bgr_10 = perform_block_segmentation(TEST_COLLAGE_PATH_10, trained_fruit_means_10, CURRENT_FRUITS, BLOCK_SIZE)
        if segmentation_map_10 is not None:
            #Visualize and Save the 10 Fruit Segmentation
            visualize_block_segmentation(segmentation_map_10, original_img_rgb_10, CURRENT_FRUITS, BLOCK_SIZE, title=f"10 Fruits Segmentation (Block Size {BLOCK_SIZE}x{BLOCK_SIZE})")
        else:
            print(f"Skipping 10-fruit visualization due to segmentation error.")
    else:
        print("Skipping 10-fruit segmentation due to no trained models.")

if __name__ == "__main__":
    main(BLOCK_SIZE=16) #Run training and segmentation for 16x16 blocks
    SimpleDecisionTree(BLOCK_SIZE=16) # Run Decision Tree classification for 16x16 blocks.