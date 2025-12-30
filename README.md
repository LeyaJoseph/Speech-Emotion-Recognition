# Speech-Emotion-Recognition
Speech Emotion Recognition (SER) aims to identify human emotions from speech signals. Emotions play a vital role in human communication and decision-making, making SER a key enabling technology for emotion-aware Human–Computer Interaction (HCI).

This project builds and evaluates deep learning models to classify emotions from speech using Mel spectrogram representations, demonstrating the effectiveness of transfer learning for audio-based emotion recognition.

---

## Motivation & Applications
SER has broad real-world applicability across multiple domains:

- **Human–Computer Interaction (HCI)**: Emotion-aware interfaces that adapt assistance or system behavior based on user state
- **Customer Service**: Automated analysis of customer support calls to assess satisfaction and emotional trends
- **Healthcare**: Supporting mental health assessment and early intervention by analyzing patient speech
- **Education**: Adaptive learning systems that gauge student engagement and tailor content accordingly
- **Other domains**: Entertainment, security, and surveillance

---

## Problem Statement
Speech Emotion Recognition is formulated as a **multi-class classification problem**, where the goal is to predict the emotional state of a speaker from audio recordings.

Challenges addressed in this project include:
- High variability in speech signals
- Limited labeled data
- Class imbalance
- Overfitting in deep learning models

---

## Dataset
**Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**

- 1,440 audio files
- 24 professional actors (12 male, 12 female)
- 8 emotion classes
- Observed class imbalance in the *neutral* class

### Data Processing & EDA
Exploratory analysis and feature extraction included:
- Mel spectrograms
- Spectral centroid
- Pitch
- RMS energy

Mel spectrograms were selected as the primary input representation for machine learning models.

---

## Approach

### Feature Representation
Audio classification was treated as an **image classification problem** by converting raw audio signals into Mel spectrograms. This enables the use of Convolutional Neural Networks (CNNs) to learn spatial patterns in the time–frequency domain.

### Class Imbalance Handling
- Random oversampling was applied to the *neutral* class to mitigate imbalance effects

---

## Models

### Model 0: CNN Trained from Scratch (Baseline)
- Convolutional layers for feature extraction
- Fully connected layers for classification
- Regularization techniques applied to mitigate overfitting

### Model 1: Fine-Tuned VGG16 + DNN
- Pre-trained VGG16 used as a feature extractor
- Custom dense neural network (DNN) classification head
- Selective fine-tuning of VGG16 layers

---

## Training Details
- Epochs: 50
- Batch size: 50
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: Sparse Categorical Cross-Entropy
- Evaluation metrics:
  - Accuracy
  - Weighted Precision
  - Weighted Recall

---

## Regularization & Generalization
Overfitting was a major challenge. The following strategies were applied:
- L2 regularization on convolutional and dense layers
- Dropout (20%) in dense layers
- Lower learning rate for stable convergence

### Data Augmentation
Audio-based augmentation techniques were explored:
- Time stretching
- Pitch shifting
- Noise injection

While augmentation increased dataset variability, it did not lead to improved model performance in this study.

---

## Hyperparameter Tuning
For the fine-tuned VGG16 model, the following hyperparameters were explored:
- Number of dense layers and neurons
- Number of trainable layers in VGG16
- Dropout rates
- Learning rate

Tuning focused on reducing overfitting while improving generalization.

---

## Results

### Model Performance Comparison

| Metric | CNN from Scratch | Fine-Tuned VGG16 |
|------|------------------|------------------|
| Accuracy (%) | 68.18 | 75.97 |
| Weighted Precision (%) | 70.21 | 76.68 |
| Weighted Recall (%) | 68.18 | 75.97 |

### Hyperparameter Tuning Outcome
- Best tuned VGG16 model achieved **79.2% accuracy**
- Lower dropout and learning rate yielded better performance
- Transfer learning significantly improved performance over training from scratch

---

## Conclusion
- Fine-tuning a pre-trained VGG16 model substantially improved SER performance over a baseline CNN
- Transfer learning helped overcome limitations posed by dataset size
- Regularization and careful hyperparameter tuning were critical to reducing overfitting
- Mel spectrograms proved effective for emotion classification from speech

---

## Repository Structure
├── EDA_Time_and_Spectral_Visualisation.ipynb

├── Feature_extraction_and_Data_Augmentation.ipynb

├── SER_models_on_Mel_Spectrograms.ipynb

├── Hyperparameter_Tuning_VGG16_Fine_Tuned_Model.ipynb

└── README.md


---

## Key Learnings
- Treating audio classification as an image problem enables effective use of CNNs
- Transfer learning is highly effective for small to medium-sized audio datasets
- Overfitting is a primary challenge in SER and requires multiple mitigation strategies
- Hyperparameter tuning can yield significant performance gains


