# Eye-Contact-AI
Eye Contact AI is currently a binary classification model for images of faces with eyes looking down (at the computer screen) and looking directly at the camera. Later, it will be able to take in video of a person looking down or to the side and readjust the eyes to point to the camera in real-time so that video meetings have more human connection.

It was trained using BinaryCrossEntropy loss and the Adam optimizer in PyTorch using Google Colab. The current dataset includes 141 new, manually-annotated images of my face. In half of the images, I looked slightly downward; in the other half, I looked directly at the camera. I used Label Studio to create color trimaps to assist with future model development, but this has not been used in the current iteration of the model.

There are significant limitations with this model. For one, there are too few images to train on. Secondly, the model was training on only images of my face, but for real-world use, diverse data of different faces in different contexts are needed. Thirdly, the accuracy of the model is only about 55%. To increase accuracy, there will be more data and more diverse data to train the model. Forth, the current model can only take in one image and classify it to 'down' and 'forward'. Fifth, there is no working user interface to interact with the model. Sixth, there is little documentation.

However, I'm working to address all of these issues so that others can collaborate on and use this model. 
