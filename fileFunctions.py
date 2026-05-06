import h5py
import os
import shutil
import cv2

def currentDirectory():
    return os.getcwd()

def saveToFile(data : dict, targetFile : str):
    '''Saves the provided data to an HDF5 file with the specified target file name. The data should be in the form of a dictionary.
    
    Parameters
    ----------
    data : dict
        The data to be saved.
    targetFile : str
        The name of the target file.
    
    '''
    
    with h5py.File(f'{targetFile}.hdf5', 'w') as hf:
        for key, value in data.items():
            hf.create_dataset(key, data=value)
    #commenting out the print statement as it was causing too much output during training, and the function is working as intended without it
    #return print(f'data added to {targetFile}.hdf5')

def clearFile(targetFile : str):
    '''Clears the contents of the specified HDF5 file erasing all data.
    
    Parameters
    ----------
    targetFile : str
        The name of the target file.
    '''
    
    with h5py.File(f'{targetFile}.hdf5', 'w') as hf:
        pass
    #commenting out the print statement as it was causing too much output during training, and the function is working as intended without it  
    #return print(f'{targetFile}.hdf5 has been cleared of data')

def loadData(targetFile : str) -> dict:
    '''Loads the data from the specified HDF5 file and returns it as a dictionary.
    
    Parameters
    ----------
    targetFile : str
        The name of the target file.
    
    Returns
    -------
    newDict : dict
        The loaded data.
    '''
    newDict = {}
    with h5py.File(f'{targetFile}.hdf5', 'r') as hf:
        for key in hf.keys():
            newDict[key] = hf[key][:]
    #commenting out the print statement as it was causing too much output during training, and the function is working as intended without it
    #return print(f'loaded data from {targetFile}.hdf5')
    return newDict

def clearFolder(folderPath : str):
    '''Clears all files from the specified folder path.
    
    Parameters
    ----------
    folderPath : str
        The path to the folder to be cleared.
    '''
    
    for file in os.listdir(folderPath):
        filePath = os.path.join(folderPath, file)
        if os.path.isfile(filePath):
            os.remove(filePath)
            print(f'{filePath} has been deleted.')

def copyFile(sourcePath : str, destinationPath : str, mode : str):
    '''Copies a file from the source path to the destination path. The mode parameter determines whether to copy a single (s) file or all (m) files in a directory.
    
    Parameters
    ----------
    sourcePath : str
        The path to the source file or directory.
    destinationPath : str
        The path to the destination file or directory.
    mode : str
        The mode of copying ('s' for single file, 'm' for multiple files).
    '''
    if mode == 'm':
        for file in os.listdir(sourcePath):
            filePath = os.path.join(sourcePath, file)
            if os.path.isfile(filePath):
                shutil.copy(filePath, destinationPath)
                print(f'{filePath} has been copied to {destinationPath}.')
    elif mode == 's':
        shutil.copy(sourcePath, destinationPath)
        print(f'{sourcePath} has been copied to {destinationPath}.')

def saveImage(image, targetPath : str):
    '''Saves the provided image to the specified target path.
    
    Parameters
    ----------
    image : numpy.ndarray
        The image to be saved.
    targetPath : str
        The path where the image will be saved.
    '''
    cv2.imwrite(targetPath, image)
    print(f'Image saved to {targetPath}')
