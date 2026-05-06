from matlabFunctions import displayRequestedImage, XTest, yTest, label
from network import initiliseNetworkValues, networkShape, predict, main, testModel
from fileFunctions import clearFolder, loadData, currentDirectory, copyFile
from imageFunctions import prepareImage
import os

def displayMenu():
    print('This neural network is designed to predict handwritten character from the Emnist dataset.')
    print('Please enter the number next to the option you would like to select: ')
    print('1. Network options')
    print('2. Make a prediction')
    print('3. View dataset')
    print('4. Exit')
    choice = input('Enter your choice: ')
    return choice

def networkMenu():
    print('Network options: ')
    print('1. View network information')
    print('2. Training options')
    choice = input('Enter your choice: ')
    return choice


def trainingMenu():
    print('Training options: ')
    print('1. Train network - this will continue training the network')
    print('2. Reset - this will reset the network values and start training from scratch YOU WILL LOSE ALL PROGRESS')
    print('3. Run test set - this will run the test set through the network and output the results to the console')
    print('4 Exit')
    choice = input('Enter your choice: ')
    return choice

def trainNetworkMenu():
    print('Current network stats: ')
    metadata = loadData('metadata')
    currentEpoch = int(metadata['epochs'][0])
    print(f'Epochs completed: {metadata["epochs"]}')
    while True:
        try:
            epochs = int(input('How many total epochs would you like to train for? (enter a number): '))
            if epochs <= currentEpoch:
                print('Please enter a number greater than the current epoch count.')
                continue
        except ValueError:
            print('Invalid input. Please enter a number greater than the current epoch count.')
        else:
            print(f'Training for {epochs} epochs...')
            print('Batach size is set to 512 and learning rate is set to 0.0001, these values cannot be changed at the moment.')
            print('To edit these values, please edit this function call and the train function call in network.py')
            print('This may take a while depending on hardware, data is saved every epoch.')
            main(512, epochs)
            break

def resetNetworkMenu():
    print('WARNING: This will reset the network values and start training from scratch, you will lose all progress.')
    print('To restore you will need to redownload networkweights.hdf5 and networkbiases.hdf5')
    while True:
        choice = input('Are you sure you want to reset the network? (y/n): ')
        if choice.lower() == 'y':
            print('Resetting network')
            L, n = networkShape()
            initiliseNetworkValues(n)
            print('Values reset, please select the train network option to start training from scratch.')
            break
        elif choice.lower() == 'n':
            print('Network reset cancelled.')
            break
        else:
            print('Invalid input. Please enter y or n.')

def viewStructure():
    L, n = networkShape()
    print(f'Network structure: {L} layers with {n} neurons each')
    print('The network is trained using the Adam optimization algorithm, categorical cross-entropy loss function,\n and ReLU activation function for the hidden layers and softmax activation function for the output layer.')
    print('The network is trained on the Emnist dataset, which contains 62 classes of handwritten characters, including digits (0-9), uppercase letters (A-Z) and lowercase letters (a-z).')
    print(r'The network achieves around 96% accuracy on the test set after 20 epochs of training, and about 85% on the test set')
    print('The cost after 20 epochs is around 0.1, the model likely will overfit if trained for longer than 20 epochs')
    print('Each epoch takes around 30 minutes to complete on a i9-10850k CPU.')

def runTestData():
    print('Running test set')
    weights = loadData('networkWeights')
    biases = loadData('networkBiases')    
    testModel(XTest, yTest, weights, biases, batchSize=1024)

while True:
    try:
        mainChoice = int(displayMenu())
        if mainChoice not in [1, 2, 3, 4]:
            print('Please enter a number from the menu options.')
            continue
    except ValueError:
        print('Invalid input. Please enter a number from the menu options.')
        continue
    if mainChoice == 1:
        while True:
            try:
                secChoice = int(networkMenu())
                if secChoice not in [1, 2]:
                    print('Please enter a number from the menu options.')
                    continue
            except ValueError:
                print('Invalid input. Please enter a number from the menu options.')
                continue
            if secChoice == 1:
                viewStructure()
                break
            elif secChoice == 2:
                while True:
                    try:
                        trainChoice = int(trainingMenu())
                        if trainChoice not in [1, 2, 3, 4]:
                            print('Please enter a number from the menu options.')
                            continue
                    except ValueError:
                        print('Invalid input. Please enter a number from the menu options.')
                        continue
                    if trainChoice == 1:
                        trainNetworkMenu()
                        break
                    elif trainChoice == 2:
                        resetNetworkMenu()
                        break
                    elif trainChoice == 3:
                        runTestData()
                        break
                    elif trainChoice == 4:
                        break
                break
            break
        
    elif mainChoice == 2:
        print('To make a prediction, please insert image(s) of a character into ..../inputFolder')
        print(r'The images will also be saved in ../networkImages/past/{orginal_file_name_predictedCharacter}.jpg')
        print('This folder can be emptied at any time')
        print('The image(s) must be in .png, .jpg or .jpeg format.')
        check = input('Enter any key once the image(s) are ready: ')
        rootPath = currentDirectory()
        predicts = []
        for file in os.listdir(f'{rootPath}/inputFolder'):
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                clearFolder(f'{rootPath}/networkImages/current')
                copyFile(f'{rootPath}/inputFolder/{file}', f'{rootPath}/networkImages/current', 's')
                X = prepareImage(f'{rootPath}/networkImages/current/{file}', f'{rootPath}/networkImages/current')
                prediction = predict(X)
                predicts.append(prediction)
                print(f'Prediction for {file}: {prediction}')
                copyFile(f'{rootPath}/networkImages/current/{file}', f'{rootPath}/networkImages/past/{file}_{prediction}.png', 's')
                copyFile(f'{rootPath}/networkImages/current/downsampledImg.png', f'{rootPath}/networkImages/past/{file}_{prediction}_standardised.png', 's')
        print('Predictions complete, please check the console for results.')
        print(f'Predictions in order of images {predicts}')
        
    elif mainChoice == 3:
        print('The Emnist dataset contains 62 classes of handwritten characters, including digits (0-9), uppercase letters (A-Z) and lowercase letters (a-z).')
        print('To view an image from the dataset, please enter the group (train or test) and the image number (0-697932 for train, 0-116323 for test).')
        while True:
            group = input('Enter group (train/test): ')
            if group not in ['train', 'test']:
                print('Invalid input. Please enter "train" or "test".')
                continue
            try:
                imageNum = int(input('Enter image number: '))
                if group == 'train' and (imageNum < 0 or imageNum >= 697932):
                    print('Please enter a number between 0 and 697932 for the train group.')
                    continue
                elif group == 'test' and (imageNum < 0 or imageNum >= 116323):
                    print('Please enter a number between 0 and 116323 for the test group.')
                    continue
            except ValueError:
                print('Invalid input. Please enter a valid number for the image.')
                continue
            displayRequestedImage(group, imageNum)
            break
           
    elif mainChoice == 4: 
        print('Exiting program')
        break
        