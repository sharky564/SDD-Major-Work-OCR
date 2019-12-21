import sys
import cv2
import editdistance
import argparse
from Loading_Data import Load_Data, Batch
from Neural_Network import Neural_Network
from Preprocess import preprocess

class paths:
    '''Paths to data'''
    fn_characters = '/Users/naveenkesarwani/Documents/Machine_Learning/SDD Task/save/characters.txt'
    fn_accuracy = '/Users/naveenkesarwani/Documents/Machine_Learning/SDD Task/save/accuracy.txt'
    fn_train = '/Users/naveenkesarwani/Documents/Machine_Learning/SDD Task/data/'
    fn_predict = '/Users/naveenkesarwani/Documents/Machine_Learning/SDD Task/data/test.png'
    fn_texts = '/Users/naveenkesarwani/Documents/Machine_Learning/SDD Task/data/texts.txt'

def train(network, loader):
    '''Train the Neural Network'''
    epoch = 0
    best_error = float('inf')
    last_improve = 0
    stopping = 5
    while True:
        epoch += 1
        print('Training the Neural Network. Beginning Epoch', epoch)
        loader.train()
        while loader.check():
            batch_info = loader.batch_info()
            next_batch = loader.next()
            cost = network.batching(next_batch)
            print('Batch', batch_info[0], '/', batch_info[1], 'trained. Current cost:', cost)

        character_error = test(network, loader)
        # output error rates and keep improving
        if character_error < best_error:
            best_error = character_error
            last_improve = 0
            network.save()
            open(paths.fn_accuracy, 'w').write('Network\'s current error rate: %f%%' % (character_error * 100))
        else:
            last_improve += 1

        if last_improve >= stopping:
            print('No improvement in the last', stopping, 'epochs. Training completed.')
            break

def test(network, loader):
    '''Test the Neural Network'''
    print('Testing the Neural Network.')
    loader.test()
    num_errors = 0
    num_errors_total = 0
    num_words = 0
    num_words_total = 0
    while loader.check():
        batch_info = loader.batch_info()
        print('Batch', batch_info[0], '/', batch_info[1], 'tested.')
        batch = loader.next()
        correct = network.test_batch(batch)

        for i in range(len(correct)):
            if batch.texts[i] == correct[i]:
                num_words += 1
            num_words_total += 1
            num_errors += editdistance.eval(correct[i], batch.texts[i])
            num_errors_total += len(batch.texts[i])

    # output error rates
    error_rate = num_errors / num_errors_total
    accuracy = num_words / num_words_total
    print('Character error rate: %f%% | Word accuracy: %f%%' % (error_rate * 100.0, accuracy * 100.0))
    return error_rate

def use(network, fn_image):
    '''recognise text in images provided by file path'''
    image = preprocess(cv2.imread(fn_image, cv2.IMREAD_GRAYSCALE), Neural_Network.image_size)
    batch1 = Batch(None, [image])
    word = network.test_batch(batch=batch1)
    print('Translation:', word)

def main():
    '''The actual program running'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='Train the Neural Network', action='store_true')
    parser.add_argument('--test', help='Test the Neural Network', action='store_true')
    parser.add_argument('--use', help='Uses and saves the outputs of the Neural Networks onto CSV files.', action='store_true')

    args = parser.parse_args()
    if args.train:
        loader = Load_Data(paths.fn_train, Neural_Network.batch_size, Neural_Network.image_size, Neural_Network.text_len)
        open(paths.fn_characters, 'w').write(str().join(loader.characters))
        open(paths.fn_texts, 'w').write(str(' ').join(loader.training_words + loader.testing_words))

        network = Neural_Network(loader.characters)
        train(network, loader)

    elif args.test:
        loader = Load_Data(paths.fn_train, Neural_Network.batch_size, Neural_Network.image_size, Neural_Network.text_len)
        open(paths.fn_characters, 'w').write(str().join(loader.characters))
        open(paths.fn_texts, 'w').write(str(' ').join(loader.training_words + loader.testing_words))
        
        network = Neural_Network(loader.characters, restore=True)
        test(network, loader)

    else:
        print(open(paths.fn_accuracy).read())
        network = Neural_Network(open(paths.fn_characters).read(), saved=args.save, restore=True)
        use(network, paths.fn_predict)

if __name__ == '__main__':
    main()
