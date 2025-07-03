# AgriGenius-Plant-Disease-Detection-with-RAG-Chatbot

This project aims to create a **real-time agricultural plant disease detection system** that identifies diseases in various plant species from images of their leaves. The system processes images using a **custom CNN with ResNet-inspired architecture** and provides detailed information and advice on detected diseases through an integrated **Retrieval-Augmented Generation (RAG) chatbot**.

![Screenshot 2025-03-19 092643](https://github.com/user-attachments/assets/89ae90c4-91d5-49c8-a4e4-1e5dd73c1800)

![Screenshot 2025-03-17 201620](https://github.com/user-attachments/assets/853434af-3d72-4a17-885e-4e0b4e4d054d)

This is the Gradio interface where users can upload plant images for disease detection.

Visual preview of the prediction screen showing disease detection in action.

---

## **Project Overview**

The AgriGenius plant disease detection system is designed to:

- Identify **multiple disease classes** across different plant species (e.g., Apple scab, Tomato late blight)
- Provide **real-time disease detection** from uploaded plant leaf images
- Offer **detailed information** about detected diseases through a conversational interface
- Suggest **treatment recommendations** and **prevention strategies** for identified plant diseases

---

## **Technologies Used**

- **PyTorch & torchvision**: For deep learning model development and training
- **NumPy & Pandas**: For data handling and manipulation
- **PIL**: For image processing
- **LangChain**: For building the RAG system
- **HuggingFaceHub**: For accessing the Mistral-7B-Instruct/any other model
- **FAISS**: For vector store implementation
- **Gradio**: For building the interactive web interface
- **Colorama**: For console styling

---

## **Key Features**

1. **Dataset Collection**  
   The project uses the New Plant Diseases Dataset from Kaggle, containing RGB images of plant leaves organized by disease classes. The dataset includes training, validation, and test sets.

2. **Data Preprocessing**  
   Images are transformed into tensors with size 256×256×3 and normalized for input to the CNN model.

3. **Model Architecture**  
   A **Custom CNN with ResNet-inspired architecture** (CNN_NeuralNet) is used for disease detection, featuring convolutional blocks with BatchNorm, ReLU activation, skip connections, MaxPooling layers, and a fully connected classifier.

4. **Training**  
   The model is trained using cross-entropy loss with the Adam optimizer and OneCycleLR scheduler. Training includes gradient clipping and weight decay for regularization.

5. **Prediction and Real-Time Recognition**  
   The trained model processes uploaded images in real-time to predict plant disease classes with high accuracy.

6. **Validation**  
   The model is validated using a pre-defined validation set with performance metrics including validation loss and accuracy.

7. **RAG Chatbot Integration**  
   An advanced feature that allows users to ask questions about detected diseases. The system uses:
   - A comprehensive knowledge base of plant disease information
   - Mistral-7B-Instruct/any other model for text generation
   - FAISS vector store with sentence-transformer embeddings for efficient information retrieval

8. **Tools and Libraries**  
   The project leverages various tools for deep learning, data handling, LLM integration, visualization, and user interface development.

9. **Real-Time Application**  
   Features a **Gradio-based interface** allowing users to upload plant images and interact with the chatbot about detected diseases.

10. **Deployment Considerations**  
    The model is saved as plant_disease_model.pth for persistence and is designed to run in a Colab environment with shared link functionality.

---

## **Model Architecture**

1. **Data Preprocessing**  
   - RGB images are transformed into tensors (256×256×3)
   - Images are normalized for consistent input to the model
  
2. **Network Structure**  
   - Custom CNN with ResNet-inspired architecture
   - Convolutional blocks with BatchNorm and ReLU activation
   - Skip connections for improved gradient flow
   - MaxPooling layers for spatial dimension reduction
   - Fully connected classifier for final prediction

3. **Training Parameters**
   - Batch Size: 32
   - Learning Rate: 0.01
   - Weight Decay: 1e-4
   - Gradient Clipping: 0.15
   - Training Epochs: 5

---

## **How to Run the Project**

1. Clone the repository
2. Install the required dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset:
   ```python
   import kagglehub
   kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
   ```
4. Train the model or use the pre-trained model:
   ```python
   # To train
   python train_model.py
   
   # To use pre-trained
   # The code will automatically load plant_disease_model.pth
   ```
5. Run the application:
   ```bash
   python app.py
   ```
6. Access the web interface and upload plant leaf images for disease detection
7. Use the chatbot to get detailed information about detected diseases

---

## **Project Structure**
```bash
- data/                           # Directory containing the plant disease dataset
- models/                         # Trained model files
  - plant_disease_model.pth       # Saved trained model
- src/                            # Source code
  - data_handling.py              # Code for data loading and preprocessing
  - model.py                      # CNN model architecture definition
  - train.py                      # Training script
  - inference.py                  # Code for making predictions
  - rag_system.py                 # RAG chatbot implementation
- app.py                          # Gradio web interface
- requirements.txt                # Required dependencies
- README.md                       # Project documentation
```
---

## **Future Scope**

1. **Mobile Application**: Develop a mobile app for farmers to use in the field without internet connectivity.
   
2. **Multi-Modal Input**: Extend the system to accept different input types (e.g., soil samples, environmental data).
   
3. **Early Detection**: Implement models to identify diseases at earlier stages before visible symptoms appear.
   
4. **Localization**: Add support for multiple languages to make the system accessible to farmers worldwide.
   
5. **Integration with IoT**: Connect with agricultural IoT devices for automated monitoring and detection.

---
