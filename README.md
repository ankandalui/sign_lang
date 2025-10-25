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

### **Core Technologies:**
- **Python 3.9+** - Primary programming language
- **OpenCV** - Computer vision and image processing
- **MediaPipe** - Hand tracking and pose estimation (21 landmarks)
- **TensorFlow/Keras** - Deep learning framework for CNN
- **Tkinter** - GUI framework for user interface

### **Machine Learning Stack:**
- **CNN (Convolutional Neural Network)** - Gesture classification
- **8-Group Classification System** - Instead of direct 26-class classification
- **Mathematical Post-processing** - Distance calculations and finger position analysis
- **MediaPipe Hand Landmarks** - 21-point hand skeleton detection

### **Additional Libraries:**
- **NumPy** - Numerical computing and array operations
- **PIL (Pillow)** - Image processing and manipulation
- **pyttsx3** - Text-to-speech conversion
- **pyenchant** - Spell checking and word suggestions
- **cvzone** - Computer vision utilities and hand detection

### **Hardware Requirements:**
- **Webcam** - For real-time hand gesture capture
- **RAM**: 4GB+ recommended
- **CPU**: Multi-core processor
- **GPU**: Optional but recommended for faster processing

## **Installation & Usage:**

### **Step 1: Install Dependencies:**
```bash
# Install all required packages
pip install -r requirements.txt

# Or install individually
pip install opencv-python numpy tensorflow keras mediapipe cvzone pyttsx3 Pillow pyenchant
```

### **Step 2: Run Applications:**
```bash
# Main GUI Application (Full Features)
python final_pred.py

# Data Collection Tool (Training Data)
python data_collection_final.py

# Command-line Prediction (Lightweight)
python prediction_wo_gui.py
```

### **Step 3: Usage Commands:**

#### **Main Application (final_pred.py):**
- **Start**: `python final_pred.py`
- **Controls**: 
  - Show hand gestures in front of camera
  - Use "Speak" button for text-to-speech
  - Use "Clear" button to reset text
  - Click suggestion buttons to correct words
- **Exit**: Close window or press Ctrl+C

#### **Data Collection (data_collection_final.py):**
- **Start**: `python data_collection_final.py`
- **Controls**:
  - Press 'n' to switch between letters (A-Z)
  - Press 'a' to start/stop recording
  - Press 'Esc' to exit
- **Output**: Saves skeleton images to AtoZ_3.1 folder

#### **Command-line Prediction (prediction_wo_gui.py):**
- **Start**: `python prediction_wo_gui.py`
- **Output**: Console shows predicted letters in real-time
- **Exit**: Press 'Esc' or Ctrl+C

## **App Flow:**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Webcam Input  │───▶│  Hand Detection   │───▶│  Landmark       │
│   (Real-time)   │    │  (MediaPipe)      │    │  Extraction     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                      │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Text-to-Speech │◀───│  Text Processing │◀───│  CNN            │
│  Output         │    │  & Spell Check   │    │  Classification │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### **Step-by-Step Process:**
1. **Input Capture**: Real-time webcam feed
2. **Hand Detection**: MediaPipe detects hand and extracts 21 landmarks
3. **Skeleton Drawing**: Draw hand skeleton on white background (400x400)
4. **CNN Classification**: 8-group gesture classification
5. **Mathematical Rules**: Post-processing to determine final letter
6. **Text Output**: Display character and build sentence
7. **Speech Output**: Convert text to speech using pyttsx3
8. **Word Suggestions**: Spell checking and correction options

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

## **Implementation Details:**

### **CNN Architecture:**
```
Input Layer: 400x400x3 (RGB skeleton image)
├── Convolutional Layer 1: 32 filters, 3x3 kernel
├── Max Pooling Layer 1: 2x2 pool size
├── Convolutional Layer 2: 64 filters, 3x3 kernel  
├── Max Pooling Layer 2: 2x2 pool size
├── Convolutional Layer 3: 128 filters, 3x3 kernel
├── Max Pooling Layer 3: 2x2 pool size
├── Flatten Layer
├── Dense Layer 1: 512 neurons
├── Dropout Layer: 0.5 rate
├── Dense Layer 2: 256 neurons
└── Output Layer: 8 neurons (groups)
```

### **Mathematical Post-Processing Algorithms:**

#### **Distance Calculation:**
```python
def distance(x, y):
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))
```

#### **Key Algorithm Rules:**

**1. Letter 'A' Detection:**
```python
if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]:
    ch1 = 'A'
```

**2. Letter 'B' Detection:**
```python
if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
    ch1 = 'B'
```

**3. Letter 'C' vs 'O' Detection:**
```python
if distance(pts[12], pts[4]) > 42:
    ch1 = 'C'
else:
    ch1 = 'O'
```

**4. Letter 'G' vs 'H' Detection:**
```python
if distance(pts[8], pts[12]) > 72:
    ch1 = 'G'
else:
    ch1 = 'H'
```

### **Hand Landmark System (21 Points):**
- **Landmarks 0-4**: Thumb (0=wrist, 1-4=thumb joints)
- **Landmarks 5-8**: Index finger (5=base, 6-8=joints)
- **Landmarks 9-12**: Middle finger (9=base, 10-12=joints)
- **Landmarks 13-16**: Ring finger (13=base, 14-16=joints)
- **Landmarks 17-20**: Pinky finger (17=base, 18-20=joints)

### **Skeleton Drawing Algorithm:**
```python
# Draw finger connections
for t in range(0, 4, 1):
    cv2.line(white, (pts[t][0]+os, pts[t][1]+os1), (pts[t+1][0]+os, pts[t+1][1]+os1), (0,255,0), 3)

# Draw palm connections
cv2.line(white, (pts[5][0]+os, pts[5][1]+os1), (pts[9][0]+os, pts[9][1]+os1), (0, 255, 0), 3)
cv2.line(white, (pts[9][0]+os, pts[9][1]+os1), (pts[13][0]+os, pts[13][1]+os1), (0, 255, 0), 3)
cv2.line(white, (pts[13][0]+os, pts[13][1]+os1), (pts[17][0]+os, pts[17][1]+os1), (0, 255, 0), 3)

# Draw landmark points
for i in range(21):
    cv2.circle(white, (pts[i][0]+os, pts[i][1]+os1), 2, (0, 0, 255), 1)
```

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