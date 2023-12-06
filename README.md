# Sports_Celeb_Classification

## Model Summary
### Chosen Model
The code implements a Convolutional Neural Network (CNN) using TensorFlow's Keras API.

### Model Architecture
- Input Layer: Conv2D layer with 32 filters, a 3x3 kernel, and ReLU activation.
- MaxPooling2D layer with a 2x2 pool size.
- Flatten layer to flatten the input for dense layers.
- Two Dense layers with 256 and 512 units, respectively, using ReLU activation.
- Dropout layer with a dropout rate of 0.5 to prevent overfitting.
- Output Dense layer with 5 units and softmax activation for multi-class classification.

### Random Seed
A random seed of 7 is set to ensure reproducibility.

## Data Preprocessing
### Dataset
The code loads images of five celebrities (Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, Virat Kohli) from specified directories and resizes them to (128, 128) pixels.

### Data Distribution
The code prints the number of images for each celebrity to provide an overview of the dataset.

### Train-Test Split
The dataset is split into training and testing sets with an 80-20 ratio, ensuring a random state for reproducibility.

### Normalization
The pixel values of the images are normalized to the range [0, 1] using TensorFlow's normalize function.

## Model Training
### Compilation
The model is compiled using the Adam optimizer and sparse categorical crossentropy loss. Accuracy is chosen as the evaluation metric.

### Training
The model is trained for 40 epochs with a batch size of 128, and 10% of the training data is used for validation.

## Model Evaluation
### Evaluation Metrics
The code evaluates the model on the test set and prints the accuracy.

### Classification Report
A classification report is generated using the classification_report function from scikit-learn, providing precision, recall, and F1-score for each class.

## Model Prediction
### Prediction Function
A function is defined to make predictions on new images. It loads an image, preprocesses it, and uses the trained model for prediction.

### Prediction Example
An example prediction is demonstrated on an image of Virat Kohli.
