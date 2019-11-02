from os import listdir


def read_files(directory):
    files = []
    for filename in listdir(directory):
        path = directory + '/' + filename
        with open(path, 'r') as file:
            text = file.read()
            files.append(text)
    return files
