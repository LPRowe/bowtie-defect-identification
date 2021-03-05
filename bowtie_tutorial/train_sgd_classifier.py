import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

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

def vis(model, X_test, y_test):
    """
    Display a random image from the test set along with the images actual and predicted labels.
    """
    index=np.random.randint(len(y_test))
    label = y_test[index]
    #prediction = model.predict(np.reshape(X_test[index], (-1, 128)))
    prediction = model.predict_proba(np.reshape(X_test[index], (-1, 128)))
    pred = np.argmax(prediction)
    confidence = prediction[0][pred]
    
    img = np.reshape(X_test[index], (16, 8))
    plt.close('all')
    plt.gray()
    plt.figure()
    plt.imshow(img)
    plt.title(f'Prediction: {MAP[pred]} | Actual: {MAP[label]}\nConfidence: {round(100*confidence, 1)}%')

if __name__ == "__main__":
    MAP = {1: "Bowtie", 0: "Nonbowtie"}
    
    # Get features (X) and labels (y)
    X, y = get_data('./training_data/data.npy')
    
    # =============================================================================
    # Split the data into a training set and a testing set (depending on the model, sometimes a validation set too)
    # A healthy split is 80% : 20% or 64% : 16% : 20% (train : validation : testing) but we will skew our
    # split towards the training set because we don't have that many samples (1500)
    # =============================================================================
    (X_train, y_train), (X_test, y_test) = split_data(X, y, test_size = 0.15)
    
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
        plt.legend()
        
    # =============================================================================
    # Train our first model
    # Notes: sgd relies on randomness during training, selecting a random_state
    #       will ensure it behaves in the same random way each time.
    #       It is a good habit to do so, since it helps with debugging and tuning hyperparameters 
    #
    #       loss function options are "hinge", "log", "modified huber" with the latter 2
    #       we can get the confidence level for each prdiction the 
    # =============================================================================
    sgd_classifier = SGDClassifier(random_state=42, loss = "modified_huber")
    sgd_classifier.fit(X_train, y_train)
    
    # =============================================================================
    # Make some predictions    
    # =============================================================================
    preds = sgd_classifier.predict(X_test)
    actual = y_test
    print('Accuracy',int(100*sum(preds == actual) / len(preds)),'%')  
    
    # =============================================================================
    # Run vis to plot a new prediction each time
    # =============================================================================
    vis(sgd_classifier, X_test, y_test)
    
    # =============================================================================
    # Save the model as a .pkl file
    # =============================================================================
