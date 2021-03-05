import pickle

import matplotlib.pyplot as plt

import numpy as np
from numpy.lib.stride_tricks import as_strided

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

def get_data(file_name):
    """
    Each row consists of comma separated values:
        standard deviation (std) of pixel values in shear 0 image, 
        standard deviation (std) of pixel values in shear 45 image,
        64 pixels values representing a flattened (8 by 8) cropped shear 0 image,
        64 pixel values representing a flattened (8 by 8) cropped shear 45 image,
        label (1 for bowtie, 0 for nonbowtie)
        
    To keep things simple, we will only use the 128 features representing the
    shear0 and shear45 pixel values as our features. 
    
    The labels will just be 1 or 0 denoting if those features belong to a bowtie or nonbowtie.
    """
    data = np.load(file_name)
    features, labels = [], []
    for row in data:
        features.append(row[2:-1])
        labels.append(row[-1])
    return np.array(features), np.array(labels)

def split_data(X_data, y_data, test_size=0.2):
    """
    Splits the data into 2 sets:
        training data: to train the model (80%)
        testing data: to test the final model on unseen data (20%)
    
    returns 2 sets of data (Taining, Testing)
    Each dataset consists of (images, labels)
    """
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size, random_state=42)
    return (X_train, y_train), (X_test, y_test)

def vis(model, X_test, y_test, shape = (16, 8)):
    """
    Display a random image from the test set along with the images actual and predicted labels.
    """
    index=np.random.randint(len(y_test))
    label = y_test[index]
    size = shape[0]*shape[1]
    
    # Some models offer prediction probability try this first
    try:
        prediction = model.predict_proba(np.reshape(X_test[index], (-1, size)))
        pred = np.argmax(prediction)
        confidence = str(round(100*prediction[0][pred], 1))+'%'
    except:
        pred = int(model.predict(np.reshape(X_test[index], (-1, size)))[0])
        confidence = 'N/A'
    
    img = np.reshape(X_test[index], shape)
    plt.close('all')
    plt.gray()
    plt.figure()
    plt.imshow(img)
    plt.title(f'Prediction: {MAP[pred]} | Actual: {MAP[label]}\nConfidence: {confidence}')
    
def evaluate(predictions, actual):
    """Prints the F1 score and accuracy of the predicted values."""
    print('Accuracy', int(100*sum(preds == actual) / len(preds)), '%')  
    print('F1', round(f1_score(actual, predictions), 2))
    print()
    
    
def max_pool(image, kernel_size = 2, stride = 2, padding = 0):
    '''
    Applies max pool to 2D image.

    image: input 2D array
    kernel_size: int, the size of the window
    stride: int, the stride of the window
    padding: int, implicit zero paddings on both sides of the input
    pool_mode: string, 'max' or 'avg'
    '''
    # reshape image to 2 dimensions (16 by 8)
    image = np.reshape(image, (16, 8))
    
    # Padding
    image = np.pad(image, padding, mode='constant')

    # Window view of A
    output_shape = ((image.shape[0] - kernel_size)//stride + 1,
                    (image.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    image_w = as_strided(image, shape = output_shape + kernel_size, 
                        strides = (stride*image.strides[0],
                                   stride*image.strides[1]) + image.strides)
    image_w = image_w.reshape(-1, *kernel_size)

    image_w = image_w.max(axis=(1,2)).reshape(output_shape)
    
    # flatten image to 1 dimension
    return np.reshape(image_w, (-1, 1))

if __name__ == "__main__":
    MAP = {1: "Bowtie", 0: "Nonbowtie"}
    
    # Get features (X) and labels (y)
    X, y = get_data('./training_data/data.npy')
    
    # =============================================================================
    # Split the data into a training set and a testing set (depending on the model, sometimes a validation set too)
    # A healthy split is 80% : 20% or 64% : 16% : 20% (train : validation : testing) but we will skew our
    # split towards the training set because we don't have that many samples (1500)
    # =============================================================================
    (X_train, y_train), (X_test, y_test) = split_data(X, y, test_size = 0.10)
    
    # =============================================================================
    # Plot the data to make sure that there are an even number of bowties and nonbowties in each 
    # dataset, scikit learn does this for us in the split_data function above but its better to be 
    # safe and check. A model trained on only nonbowties wouldn't be a very good model.
    # =============================================================================
    show_data_dist = True
    if show_data_dist:
        plt.close('all')
        plt.figure() # adjust or delete dpi (dots per inch) if figure size is not right
        bar_width = 0.3
        plt.hist(y_train-bar_width, label = 'train', rwidth=bar_width)
        plt.hist(y_test, label = 'test', rwidth=bar_width)
        plt.xlabel('Left: Non-bowties ; Right: Bowties')
        plt.ylabel('Count')
        plt.xticks([])
        plt.legend()
        
    # =============================================================================
    # Train our first model
    # Notes: sgd relies on randomness during training, selecting a random_state
    #       will ensure it behaves in the same random way each time.
    #       It is a good habit to do so, since it helps with debugging and tuning hyperparameters 
    #
    #       For SGDClassifier loss function options are "hinge", "log", "modified huber" with 
    #       the latter 2 we can get the confidence level for each prediction the 
    #       
    #       We can change between models very easily, however different models will
    #       require different preprocessing steps and have different hyperparameters
    #       to tune.  We won't get into this today but you can try replacing the
    #       model with:
    #                   RidgeClassifier(random_state=42)
    #                   PassiveAggressiveClassifier(random_state=42)
    #                   DecisionTreeClassifier(random_state=42)
    # =============================================================================
    model = SGDClassifier(random_state=42, loss = "modified_huber")
    model.fit(X_train, y_train)
    
    # =============================================================================
    # Save model for future use    
    # Note: model can be reloaded later using pickle.load(open(model_name, 'rb'))
    #       note new data must be processed exactly the same as the data used to
    #       train the model for the model to work well
    # =============================================================================
    model_name = './trained_classifiers/sgd_classifier.pkl'
    pickle.dump(model, open(model_name, 'wb'))

    # =============================================================================
    # Make some predictions    
    # Notes: Accuracy here is measured as the number of correct predictions over
    #        the number of total predictions made.  However, accuracy is not always
    #        the best metric to judge your model.  A more telling metric is the F1
    #        score which is based on the precision and recall of the model. It is
    #        similar to accuracy in that 1.0 is a perfect model.
    #        Read more about F1 score here: https://en.wikipedia.org/wiki/F-score
    # =============================================================================
    preds = model.predict(X_test)
    actual = y_test
    print("SGD Classifier")
    evaluate(preds, actual) # Acc. 67%, F1 0.74
    
    # =============================================================================
    # Run vis to plot a new prediction each time
    # =============================================================================
    vis(model, X_test, y_test)
    
    # =============================================================================
    # Most models tend to perform better when features similarly scaled. (Decision trees are an exception.)
    # =============================================================================
    
    # Scale features
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Fit and test the model
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("SGD Classifier with Standard Scaling")
    evaluate(preds, actual) # Acc. 76%, F1 0.77
    
    # =============================================================================
    # Finally many models suffer from "the curse of dimensionality" meaning too many
    # data points per sample actually hurts the models performance, it's better to choose the
    # features that are most important.
    #
    # One popular method is called max-pooling, only taking the most intense pixel
    # from each pool of say 4 pixels. 
    # =============================================================================
    
    # Apply max pooling to the training and test samples
    X_train = np.array([max_pool(sample) for sample in X_train])
    X_train = np.reshape(X_train, X_train.shape[:2])
    X_test = np.array([max_pool(sample) for sample in X_test])
    X_test = np.reshape(X_test, X_test.shape[:2])
    
    # Fit and test the model
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("SGD Classifier with Standard Scaling and Max Pooling")
    evaluate(preds, actual) # Acc. 79%, F1 0.8
    
    vis(model, X_test, y_test, shape = (8, 4))