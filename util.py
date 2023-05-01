import os

def get_words_in_file(filename):
    """ Returns a list of all words in the file at filename. """
    with open(filename, 'r', encoding = "ISO-8859-1") as f:
        # read() reads in a string from a file pointer, and split() splits a
        # string into words based on whitespace
        words = f.read().split()
    return words

def get_files_in_folder(folder):
    """ Returns a list of files in folder (including the path to the file) """
    filenames = os.listdir(folder)
    # os.path.join combines paths while dealing with /s and \s appropriately
    full_filenames = [os.path.join(folder, filename) for filename in filenames]
    return full_filenames

def get_counts(file_list):
    """ 
    Returns a dict whose keys are words and whose values are the number of 
    files in file_list the key occurred in. 
    """
    counts = Counter()
    for f in file_list:
        words = get_words_in_file(f)
        for w in set(words):
            counts[w] += 1
    return counts

def get_word_freq(file_list):
    """ 
    Returns a dict whose keys are words and whose values are word freq
    """
    counts = Counter()
    for f in file_list:
        words = get_words_in_file(f)
        for w in words:
            counts[w] += 1
    return counts

class Counter(dict):
    """
    Like a dict, but returns 0 if the key isn't found.
    """
    def __missing__(self, key):
        return 0

