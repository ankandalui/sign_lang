# Sign-Language-To-Text-and-Speech-Conversion

**ABSTRACT:** 

Sign language is one of the oldest and most natural form of language for communication, hence we have come up with a real time method using neural networks for finger spelling based American sign language. Automatic human gesture recognition from camera images is an interesting topic for developing vision. We propose a convolution neural network (CNN) method to recognize hand gestures of human actions from an image captured by camera. The purpose is to recognize hand gestures of human task activities from a camera image. The position of hand and orientation are applied to obtain the training and testing data for the CNN. The hand is first passed through a filter and after the filter is applied where the hand is passed through a classifier which predicts the class of the hand gestures. Then the calibrated images are used to train CNN. 

## **Introduction:**

American sign language is a predominant sign language Since the only disability D&M people have been communication related and they cannot use spoken languages hence the only way for them to communicate is through sign language. Communication is the process of exchange of thoughts and messages in various ways such as speech, signals, behavior and visuals. Deaf and dumb(D&M) people make use of their hands to express different gestures to express their ideas with other people. Gestures are the nonverbally exchanged messages and these gestures are understood with vision. This nonverbal communication of deaf and dumb people is called sign language. 

In our project we basically focus on producing a model which can recognise Fingerspelling based hand gestures in order to form a complete word by combining each gesture. The gestures we aim to train are as given in the image below. 

## **Requirements:**

More than 70 million deaf people around the world use sign languages to communicate. Sign language allows them to learn, work, access services, and be included in the communities.  

It is hard to make everybody learn the use of sign language with the goal of ensuring that people with disabilities can enjoy their rights on an equal basis with others. 

So, the aim is to develop a user-friendly human computer interface (HCI) where the computer understands the American sign language This Project will help the dumb and deaf people by making their life easy. 

## **Objective:**
To create a computer software and train a model using CNN which takes an image of hand gesture of American Sign Language and shows the output of the particular sign language in text format converts it into audio format. 

## **Scope:**
This System will be Beneficial for Both Dumb/Deaf People and the People Who do not understands the Sign Language. They just need to do that with sign Language gestures and this system will identify what he/she is trying to say after identification it gives the output in the form of Text as well as Speech format. 

## **Tech Stack:**

**Hardware Requirement:**
- Webcam 

**Software Requirement:**
- Operating System: Windows 8 and Above 
- IDE: PyCharm 
- Programming Language: Python 3.9+
- Python libraries: OpenCV, NumPy, Keras, MediaPipe, TensorFlow, cvzone, pyttsx3, pyenchant

## **Installation & Usage:**

### **Install Dependencies:**
```bash
pip install -r requirements.txt
```

### **Run Applications:**
```bash
# Main GUI Application
python final_pred.py

# Data Collection Tool  
python data_collection_final.py

# Command-line Prediction
python prediction_wo_gui.py
```

## **How It Works:**

### **Data Acquisition:**
The system uses vision-based methods where the computer webcam is the input device for observing hand information. The main challenge of vision-based hand detection ranges from coping with the large variability of the human hand's appearance due to a huge number of hand movements, to different skin-color possibilities as well as to the variations in viewpoints, scales, and speed of the camera capturing the scene.

### **Data Pre-processing and Feature Extraction:**
In this approach for hand detection, firstly we detect hand from image that is acquired by webcam and for detecting a hand we used media pipe library which is used for image processing. So, after finding the hand from image we get the region of interest (Roi) then we cropped that image and convert the image to gray image using OpenCV library after we applied the gaussian blur. The filter can be easily applied using open computer vision library also known as OpenCV. Then we converted the gray image to binary image using threshold and Adaptive threshold methods.

We have collected images of different signs of different angles for sign letter A to Z.

### **Gesture Classification - CNN:**

**Convolutional Neural Network (CNN)**
CNN is a class of neural networks that are highly useful in solving computer vision problems. They found inspiration from the actual perception of vision that takes place in the visual cortex of our brain. They make use of a filter/kernel to scan through the entire pixel values of the image and make computations by setting appropriate weights to enable detection of a specific feature.

**8-Group Classification System:**
Because we got bad accuracy in 26 different classes thus, We divided whole 26 different alphabets into 8 classes in which every class contains similar alphabets:
- [y,j] 
- [c,o] 
- [g,h] 
- [b,d,f,I,u,v,k,r,w] 
- [p,q,z] 
- [a,e,m,n,s,t]

All the gesture labels will be assigned with a probability. The label with the highest probability will treated to be the predicted label.

So when model will classify [aemnst] in one single class using mathematical operation on hand landmarks we will classify further into single alphabet a or e or m or n or s or t.

Finally, we got **97%** Accuracy (with and without clean background and proper lightning conditions) through our method. And if the background is clear and there is good lightning condition then we got even **99%** accurate results.

### **Text To Speech Translation:**
The model translates known gestures into words. we have used pyttsx3 library to convert the recognized words into the appropriate speech. The text-to-speech output is a simple workaround, but it's a useful feature because it simulates a real-life dialogue.

## **Project Files:**

- `final_pred.py` - Main GUI application with full features
- `data_collection_final.py` - Training data collection tool
- `prediction_wo_gui.py` - Command-line prediction without GUI
- `cnn8grps_rad1_model.h5` - Pre-trained CNN model
- `requirements.txt` - Python dependencies
- `AtoZ_3.1/` - Training dataset (180 images per letter A-Z)

## **Usage Instructions:**

### **Main Application (final_pred.py):**
- Real-time hand gesture recognition with GUI
- Text display and speech output
- Word suggestions and spell checking
- Interactive buttons for word correction

### **Data Collection (data_collection_final.py):**
- Press 'n' to switch between letters (A-Z)
- Press 'a' to start/stop recording
- Press 'Esc' to exit

### **Command-line Prediction (prediction_wo_gui.py):**
- Lightweight recognition without GUI
- Console output of predicted letters

## **Performance:**
- **Accuracy**: 97% (general conditions), 99% (optimal conditions)
- **Real-time Processing**: ~30 FPS
- **Model Size**: ~50MB
- **Input Resolution**: 400x400 pixels

## **Troubleshooting:**

**Common Issues:**
1. **ModuleNotFoundError: No module named 'mediapipe'**
   ```bash
   pip install mediapipe
   ```

2. **FileNotFoundError: cnn8grps_rad1_model.h5**
   - Ensure model file is in project directory

3. **OpenCV version conflicts**
   ```bash
   pip install opencv-python>=4.6.0
   ```

4. **Camera not detected**
   - Check webcam permissions
   - Verify camera is not used by other applications

## **System Requirements:**
- **RAM**: 4GB+ recommended
- **CPU**: Multi-core processor  
- **GPU**: Optional but recommended
- **Storage**: 1GB+ for model and dependencies
- **Webcam**: Required for real-time recognition