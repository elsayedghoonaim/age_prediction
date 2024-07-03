# Age Prediction from Facial Images

## Project Overview

This project aims to predict a person's age from facial images using a Convolutional Neural Network (CNN). The model is trained on a dataset of facial images with corresponding age labels.

## Dataset

The dataset used is 'age_gender.csv', which contains:
- 23,705 entries
- Features include age, ethnicity, gender, image name, and pixel data
- Images are 48x48 pixels in grayscale

## Dependencies

- numpy
- pandas
- scikit-learn
- tensorflow / keras
- matplotlib

## Project Structure

1. Data Import and Exploration
2. Data Preprocessing
   - Image handling (converting pixel data to image arrays)
   - Feature scaling (StandardScaler for age)
3. Data Splitting (Train/Validation/Test)
4. Model Architecture
   - CNN with multiple convolutional and pooling layers
   - Dropout for regularization
   - Dense layers for final prediction
5. Model Training and Evaluation

## Model Architecture

The CNN architecture consists of:
- Input shape: (48, 48, 1)
- 3 Convolutional layers with ReLU activation
- MaxPooling after each convolutional layer
- Dropout (0.7) for regularization
- Flatten layer
- Dense layer with 512 units and ReLU activation
- Output layer with 1 unit (age prediction)

## Training

- Optimizer: Adam (learning rate = 0.001)
- Loss function: Mean Squared Error
- Metric: Mean Absolute Error
- Batch size: 64
- Epochs: 10

## Results

- The model achieves an R2 score of approximately 0.79 on the test set.
- Training and validation loss curves are plotted to visualize model performance.

## Future Improvements

- Fine-tune hyperparameters
- Experiment with different model architectures
- Implement data augmentation techniques
- Explore transfer learning with pre-trained models

## Usage

[Include instructions on how to run the code, if any specific setup is required]

## Contributors

[Your Name/Team]

## License

[Specify the license under which this project is released]
