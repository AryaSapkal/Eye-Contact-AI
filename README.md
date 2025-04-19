# Eye-Contact-AI
Eye Contact AI is currently a binary classification model for images of faces with eyes looking down (at the computer screen) and looking directly at the camera. Later, it will be able to take in video of a person looking down or to the side and readjust the eyes to point to the camera in real-time so that video meetings have more human connection.

It was trained using BinaryCrossEntropy loss and the Adam optimizer in PyTorch using Google Colab. The current dataset includes 141 new, manually-annotated images of my face. In half of the images, I looked slightly downward; in the other half, I looked directly at the camera. I used Label Studio to create color trimaps to assist with future model development, but this has not been used in the current iteration of the model.

There are significant limitations with this model. For one, there are too few images to train on. Secondly, the model was training on only images of my face, but for real-world use, diverse data of different faces in different contexts are needed. Thirdly, the accuracy of the model is only about 55%. To increase accuracy, there will be more data and more diverse data to train the model. Forth, the current model can only take in one image and classify it to 'down' and 'forward'. Fifth, there is no working user interface to interact with the model. Sixth, there is little documentation.

However, I'm working to address all of these issues so that others can collaborate on and use this model. 

Motivation

My goal with this project is to learn how to create a basic AI-based application. While I took a course named “Concepts of Artificial Intelligence” in college, I started learning PyTorch through Daniel Bourke’s (mrdbourke) YouTube course on machine learning and deep learning with PyTorch. I wanted to create a new app to improve my skills in various parts of AI development while solving a problem I often saw and experienced myself.

In video conference meetings, people look at the center of their laptop screen and find it very challenging to look at the camera at the top when speaking. Looking directly at the camera improves professional relationships but is impractical because it doesn’t allow the speaker to see the listener’s facial expressions while they’re speaking. Most tools also make it difficult for a user to move the listener’s video frame near their laptop camera. With my project, eye contact can be maintained as if there was a camera at the center of the screen. This is a problem simple enough for me to tackle, so I proceeded with this.

By the end of this project, I hope to learn three main things: 1) how to create a complete AI-powered app, starting from creating new data and performing data analysis and preparation all the way through to implementing the user experience so that the user can interact with an AI model; 2) how neural networks work on a deep level; and 3) how to solve a real-world problem.



Tools I Used

For this project, I used these tools:
Python
PyTorch
NumPy, Matplotlib
Google Colab Notebook
Label Studio



Model & Dataset

I created a binary classification model based on a convolution neural network (CNN) architecture that takes in 1920-pixel by 1080-pixel images of a face and outputs where the eyes are looking “down” (center of the screen) or “forward” (directly at the camera). While this is a work in progress, I intend to move from binary classification to image-to-image translation. I intend to experiment with 1) attention mechanisms; 2) a pipeline to a) detect the region around the eyes and mark it (to focus the model on a specific area) and b) shift the eyes to a position making eye contact with the camera; and 3) generative adversarial networks (GANs). 

The 141-image dataset includes:
60 training images with eyes looking forward
15 testing images with eyes looking forward
50 training images with eyes looking down
16 testing images with eyes looking down


Training Process

Input size: 1920 by 1080
Transformed size: [256,256,3]



Permuted to [3,256,256] to work with PyTorch’s color channels-first approach







SimpleEyeModel:
Convolution block 1: nn.Conv2d, nn.ReLU, nn.Conv2d, nn.ReLU, nn.MaxPool2d


Layer
Output Shape (if input = 3×256×256)
Input
(B, 3, 256, 256)
conv_block_1
(B, 10, 128, 128)
conv_block_2
(B, 10, 64, 64)
Flatten
(B, 10 × 64 × 64 = 40960)



Layer
Shape In
Shape Out
Flatten
(B, 10, 64, 64)
(B, 40960)
Linear 1
(B, 40960)
(B, 1024)
ReLU
(B, 1024)
(B, 1024)
Linear 2
(B, 1024)
(B, 2)
Tanh
(B, 2)
(B, 2)




Input shape: 3
Hidden units: 10
Loss function: nn.MSELoss
Optimizer: Adam optimizer

model = SimpleEyeModel(input_shape=3, # number of color channels (3 for RGB)
                  hidden_units=10,
                  output_shape=len(train_data.classes)).to(device)  # Move model to GPU if available
criterion = nn.MSELoss()  # Example loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)


A batch size of 32 was used in the DataLoader. This means 4 batches of 32 tensors each and 1 batch with 13 images.




Current Results
![Model Results for Eye Contact Model v  4 18 2025](https://github.com/user-attachments/assets/8bc655d2-ad5f-47cf-a62e-efd428e43de7)



Model summary:
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
SimpleEyeModel                           [1, 2]                    --
├─Sequential: 1-1                        [1, 10, 128, 128]         --
│    └─Conv2d: 2-1                       [1, 10, 256, 256]         280
│    └─ReLU: 2-2                         [1, 10, 256, 256]         --
│    └─Conv2d: 2-3                       [1, 10, 256, 256]         910
│    └─ReLU: 2-4                         [1, 10, 256, 256]         --
│    └─MaxPool2d: 2-5                    [1, 10, 128, 128]         --
├─Sequential: 1-2                        [1, 10, 64, 64]           --
│    └─Conv2d: 2-6                       [1, 10, 128, 128]         910
│    └─ReLU: 2-7                         [1, 10, 128, 128]         --
│    └─Conv2d: 2-8                       [1, 10, 128, 128]         910
│    └─ReLU: 2-9                         [1, 10, 128, 128]         --
│    └─MaxPool2d: 2-10                   [1, 10, 64, 64]           --
├─Sequential: 1-3                        [1, 2]                    --
│    └─Flatten: 2-11                     [1, 40960]                --
│    └─Linear: 2-12                      [1, 1024]                 41,944,064
│    └─ReLU: 2-13                        [1, 1024]                 --
│    └─Linear: 2-14                      [1, 2]                    2,050
│    └─Tanh: 2-15                        [1, 2]                    --
==========================================================================================
Total params: 41,949,124
Trainable params: 41,949,124
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 149.75
==========================================================================================
Input size (MB): 0.79
Forward/backward pass size (MB): 13.12
Params size (MB): 167.80
Estimated Total Size (MB): 181.70
==========================================================================================




Limitations
Low amount of images
There are two few images in the dataset to create a model with at least 80% accuracy. In the future, I plan to use existing eye-region datasets to improve the models.
Non-diverse images
Since the data was trained only on my face, it would perform worse on other faces. This will be solved with images of different faces in different contexts.
CNN takes in all features instead of only relevant features
To improve the accuracy of the model while reducing compute costs, I aim to experiment with ways to focus the model on the eye region specifically while mainly ignoring other parts of the image.


What I Learned
Data type, data shape, and device are extremely important
How to quickly experiment before writing functions
How to use a notebook
How to organize/preprocess data
How to visualize model results with Matplotlib



Next Steps and Improvements
Learn how to integrate transfer learning to significantly boost 
Collect new data from various sources to improve model accuracy and reduce bias in the dataset
Improve the accuracy by experimenting 


How to Run the Code

Currently, the code resides in this Google Colab notebook: https://colab.research.google.com/drive/10Ju-F71ra3QQYTePfRgFFCwmXsHN_yN9?usp=sharing


