import cv2
import numpy as np
import os
import random
from Preprocess import preprocess


class Sample:
    '''Extract a sample from the dataset'''

    def __init__(self, text, path):
        self.text = text
        self.path = path


class Batch:
    '''Batch for the use of training'''

    def __init__(self, texts, images):
        self.images = np.stack(images, axis=0)
        self.texts = texts


class Load_Data:
    '''Loads in data'''

    def __init__(self, path, batch_size, image_size, text_len):
        '''Load dataset from given location, preprocess images and text'''

        assert path[-1] == '/'
        self.randomiser = False
        self.current = 0
        self.batch_size = batch_size
        self.image_size = image_size
        self.samples = []

        f = open(path + 'words.txt')
        chars = set()
        # find corrupted files
        corrupted = []
        known_corrupted = ['a01-117-05-02.png', 'r06-022-03-05.png']
        # isolate the words from each line and compile them in a new text file
        for line in f:
            if not line or line[0] == '#':
                continue

            line_split = line.strip().split(' ')
            assert len(line_split) >= 9

            filename_split = line_split[0].split('-')
            filename = path + 'words/' + \
                filename_split[0] + '/' + filename_split[0] + '-' + filename_split[1] + '/' + line_split[0] + '.png'

            word = self.formatted(' '.join(line_split[8:]), text_len)
            chars = chars.union(set(list(word)))

            if not os.path.getsize(filename):
                corrupted.append(line_split[0] + '.png')
                continue

            self.samples.append(Sample(word, filename))

        # give warning that there are some additional corrupted files found
        if set(corrupted) != set(known_corrupted):
            print('Warning - Corrupted files found in dataset:', corrupted)
            print('Expected corrupted files:', known_corrupted)

        # splitting the dataset into either training or testing
        sample_split = int(0.95 * len(self.samples))
        self.training_samples = self.samples[:sample_split]
        self.testing_samples = self.samples[sample_split:]

        self.training_words = [y.text for y in self.training_samples]
        self.testing_words = [y.text for y in self.testing_samples]

        self.num_samples = 25000

        self.train()
        # compiling a list of characters
        self.characters = sorted(list(chars))

    def formatted(self, text, text_len):
        '''The Connectionist Temporal Classification requires the words to be modified when being scored because of letter repeats'''
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > text_len:
                return text[:i]
        return text

    def train(self):
        '''Choosing subset of data used for training'''
        self.randomiser = True
        self.current = 0
        random.shuffle(self.training_samples)
        self.samples = self.training_samples[:self.num_samples]

    def test(self):
        '''Choosing subset of data used for testing'''
        self.randomiser = False
        self.current = 0
        self.samples = self.testing_samples

    def batch_info(self):
        '''Number of batches trained through, as well as total number of batches'''
        return (self.current // self.batch_size + 1,
                len(self.samples) // self.batch_size)

    def check(self):
        '''Checks if next batch is possible'''
        return self.current + self.batch_size <= len(self.samples)

    def next(self):
        '''Goes to next batch'''
        batch_range = range(self.current, self.current + self.batch_size)
        texts = [self.samples[i].text for i in batch_range]
        images = [
            preprocess(
                cv2.imread(
                    self.samples[i].path,
                    cv2.IMREAD_GRAYSCALE),
                self.image_size,
                self.randomiser) for i in batch_range]
        self.current += self.batch_size
        return Batch(texts, images)
