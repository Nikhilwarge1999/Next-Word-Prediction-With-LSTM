# Next Word Prediction with LSTM and Early Stopping

## Overview
This repository implements a next-word prediction model using Long Short-Term Memory (LSTM) networks. The model is trained on Shakespeare's *Hamlet* text and utilizes early stopping to optimize training performance.

## Features
- **Dataset**: Uses `shakespeare-hamlet.txt` as the training dataset.
- **Model Architecture**: The model is built using an Embedding layer, an LSTM layer, and a Dense layer.
- **Training Optimization**: Early stopping is implemented to prevent overfitting and reduce training time.
- **Prediction**: Generates the next word based on the input sequence provided by the user.

## Dataset
The dataset used for this project is the text from *Hamlet* by William Shakespeare. It is stored as a plain text file named `shakespeare-hamlet.txt`. The file is preprocessed to tokenize the text and create input-output sequences for training the model.

## Requirements
The project requires the following Python libraries:

- TensorFlow
- NumPy
- Requests
- Keras (comes with TensorFlow)

To install the required dependencies, run:


## How to Use

1. **Clone the Repository**
   

2. **Prepare the Dataset**
   Ensure that `shakespeare-hamlet.txt` is located in the repository directory or downloaded during runtime.

3. **Train the Model**
   Run the training script to preprocess the data, train the model, and save it:
   

4. **Predict the Next Word**
   Use the saved model to predict the next word for a given input sequence:
   

## Deployment on Streamlit
Follow these steps to deploy the model using Streamlit:

1. **Install Streamlit**
   Install Streamlit if not already installed:
   

2. **Create the Streamlit App**
   Create a file named `app.py` in the repository directory with the following content:
   

3. **Run the Streamlit App**
   Launch the app locally:
   

4. **Access the App**
   Open the URL displayed in the terminal (usually `http://localhost:8501`) to access the Streamlit app.

## Model Details
- **Embedding Layer**: Maps words to dense vector representations.
- **LSTM Layer**: Captures temporal dependencies in the text data.
- **Dense Layer**: Predicts the next word using a softmax activation function.
- **Early Stopping**: Monitors validation loss and halts training if no improvement is observed for a specified number of epochs.

## Results
The model achieves reasonable accuracy in predicting the next word based on Shakespearean text. Predictions improve as the length of input sequences increases.

## Contributing
Feel free to fork the repository and submit pull requests for improvements or bug fixes. Contributions are always welcome!

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- [TensorFlow](https://www.tensorflow.org/)
- William Shakespeare's *Hamlet* for providing the dataset.

