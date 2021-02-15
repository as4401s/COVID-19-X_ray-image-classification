"""
@author: asarkar

Contains various helper functions for the original script
"""

import matplotlib.pyplot as plt
import random
import cv2 as cv
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.applications.efficientnet import preprocess_input

def class_counter(data_labels):
    """
    This function counts and returns the number of images in each class
    """
    labels = ['1_Normal', '2_Non_Covid_19','3_Covid_19']

    normal = 0
    non_covid_19 = 0
    covid_19 = 0

    for i in data_labels:
        
        if i==0:
            normal += 1
        elif i==1:
            non_covid_19 += 1
        else:
            covid_19 += 1
    
    return normal,non_covid_19,covid_19
    
# plotting the data
def display_bar_chart(data):
    """
    This function plots the number of images in a class as a bar chart
    """
    data_labels = data
    normal,non_covid_19,covid_19 = class_counter(data_labels)
    labels = ['1_Normal', '2_Non_Covid_19','3_Covid_19']
    xe = [i for i, _ in enumerate(labels)]

    numbers = [normal, non_covid_19, covid_19]
    colors = ['green','blue','red']
    plt.bar(xe,numbers,color = colors)
    plt.xlabel("Labels")
    plt.ylabel("No. of images")
    plt.title("Images for each label")

    plt.xticks(xe, labels)
    plt.show()

def display_images(dataset):
    """
    This function displays 9 random images in a dataset 
    """
    # Extract 9 random images
    print('Display Random Images')

    # Adjust the size of your images
    plt.figure(figsize=(20,10))

    for i in range(9):
        num = random.randint(0,len(dataset)-1)
        plt.subplot(3, 3, i + 1)
        plt.imshow(dataset[num],cmap='gray')
        plt.axis('off')
    # Adjust subplot parameters to give specified padding
    plt.tight_layout()
    
def get_confusion_matrix_stats(y_actual, y_pred): 
    """
    This function calculates and returns the True Positive, True negative,
    False Positive and False negative values,
    and returns them
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if y_actual[i]==y_pred[i]==1:
            TP += 1
        if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
            FP += 1
        if y_actual[i]==y_pred[i]==0:
            TN += 1
        if y_pred[i]==0 and y_actual[i]!=y_pred[i]:
            FN += 1

    return(TP, FP, TN, FN)

def calculate_metrics(y_true,y_predicted,name):
    """
    This function calculates the Precision, Recall and F-score and returns them
    """
    y_actual = y_true
    y_pred = y_predicted
    tp, fp, tn ,fn = get_confusion_matrix_stats(y_actual, y_pred)

    precision = round((tp/(tp+fp)),2)
    recall = round((tp/(tp+fn)),2)
    f_score = round(((2*precision*recall)/(precision+recall)),2)

    print(f"Recall of {name} is = ",recall)
    print(f"Precision of {name} is = ",precision)
    print(f"F-Score is {name} is =",f_score)
    print()
    
def show_image_and_prediction(num,model,X_test,y_test,image_size=380):
    """
    This function takes an image from the test set, preprocesses it,
    predicts the class label of the image,
    calculates the accuracy,
    displays the image and the prediction score of the image as a bar chart
    """
    #prepocess the image
    image = X_test[num]
    image = cv.resize(image,(image_size,image_size),cv.IMREAD_GRAYSCALE)
    display = cv.resize(image,(720,720))
    image = preprocess_input(image)
    image = np.expand_dims(image,axis=0)
    
    #predict the class
    preds = model.predict(image)
    
    #calculate accuracy
    accuracy = np.round(np.max(preds)*100,2)
    i = int(preds.argmax(axis=1))
    scores = preds[0]*100
    
    #class labels
    names = ['Normal','Non-COVID-19','COVID-19']
    predicted_label = names[i]
    #color of bar chart
    colors = ['green','blue','crimson']
    #plotting the prediction scores
    fig = go.Figure(data=[go.Bar(x=names,y=scores,text=scores,textposition='auto',marker_color=colors)])
    fig.update_layout(title_text='Prediction Percentage (%)')
    to_image = fig.to_image(format="png",width=720, height=720)
    final = cv.imdecode(np.frombuffer(to_image, np.uint8), 1)
    
    #original label
    if y_test[num] == 0:
        original_label = 'Normal'
                
    if y_test[num] == 1:
        original_label = 'Non-COVID-19'
                
    if y_test[num] ==2:
        original_label = 'COVID-19'
    
    #write text on the predicted image
    font                   = cv.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (150,700)
    bottomLeftCornerOfText2 = (10,700)
    fontScale              = 0.8
    fontColor              = (120,5,5)
    lineType               = 2
    
    #put text on image and bar chart
    cv.putText(display,f'Original Label = {original_label}',bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    cv.putText(final,f'Predicted Label = {predicted_label}, Accuracy = {accuracy} %', 
               bottomLeftCornerOfText2, font, fontScale, fontColor, lineType)
    
    #print original label, predicted label and accuracy
    print('Original Label = ',original_label)
    print('Predicted Label = ',predicted_label)
    print('Accuracy of prediction in % = ',accuracy)
    
    #display the image and the bar chart side by side
    fig = plt.figure(figsize=(15,15))
    plt.subplot(1, 2, 1)
    plt.imshow(display)

    plt.subplot(1, 2, 2)
    plt.imshow(final)
    
    plt.show()