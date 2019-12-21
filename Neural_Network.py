import cv2
import tensorflow as tf
import os

class Neural_Network:

    # constants for the network
    image_size = [128, 32]
    text_len = 32
    batch_size = 50

    def __init__(self, characters, saved=False, restore=False):

        self.saved = saved
        self.characters = characters
        self.restore = restore
        self.current = 0

        # inputting the images in batches
        self.batches = tf.placeholder(tf.float32, shape=(None, Neural_Network.image_size[0], Neural_Network.image_size[1]))

        # setting up the required neural networks
        self.Convolutional_Neural_Network()
        self.Recurrent_Neural_Network()
        self.Connectionist_Temporal_Classification()

        # setting up optimiser
        self.progress = 0
        self.rate = tf.placeholder(tf.float32, shape=[])
        self.updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.updates):
            self.optimiser = tf.train.AdamOptimizer(self.rate).minimize(self.cost)

        (self.sess, self.saver) = self.setup()

    def setup(self):
        '''Necessary things required for TensorFlow to run'''
        # TF session
        sess = tf.Session()

        saver = tf.train.Saver(max_to_keep=1)
        save_dir = '/Users/naveenkesarwani/Documents/Machine_Learning/SDD Task/save/'
        latest = tf.train.latest_checkpoint(save_dir)

        if self.restore and not latest:
            raise Exception('No network currently saved')

        if latest:
            print('Network', latest, 'found. Proceeding to initiate with stored values.')
            saver.restore(sess, latest)
        else:
            print('No network found. Proceeding to initiate with new values.')
            sess.run(tf.global_variables_initializer())

        return (sess, saver)

    def Convolutional_Neural_Network(self):
        '''Create the Convolutional Neural Network layers and convert the image from 128 x 32 to 32 x 256'''
        conv4d_in = tf.expand_dims(input=self.batches, axis=3)

        # constants for the layers
        weight_values = [5, 5, 3, 3, 3]
        feature_values = [1, 32, 64, 128, 128, 256]
        stride_values = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        pool_values = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        number_layers = 5

        # create layers
        data = conv4d_in
        for layer in range(number_layers):
            weight = tf.Variable(tf.random_normal([weight_values[layer], weight_values[layer], feature_values[layer], feature_values[layer + 1]]))
            conv = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(data, weight, strides=(1, 1, 1, 1), padding='SAME')))
            data = tf.nn.max_pool(conv, (1, pool_values[layer][0], pool_values[layer][1], 1), (1, stride_values[layer][0], stride_values[layer][1],1), padding='VALID')

        self.conv4d_out = data

    def Recurrent_Neural_Network(self):
        '''Create the Recurrent Neural Network layers and create the score of the 32 x 256 image on a 32 (number of time-steps) x 80 (number of characters) matrix'''
        rec3d_in = tf.squeeze(self.conv4d_out, axis=[2])

        # LSTM Cell for the network
        rnn_size = 256
        lstm_cells = [tf.nn.rnn_cell.LSTMCell(rnn_size) for _ in range(2)]

        # stack cells
        stack = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

        # construct information flow in both directions
        ((forward, backward), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stack, cell_bw=stack, inputs=rec3d_in, dtype=rec3d_in.dtype)
        data = tf.expand_dims(tf.concat([forward, backward], 2), 2)

        # send output to characters
        weight = tf.Variable(tf.random_normal([1, 1, rnn_size * 2, len(self.characters) + 1]))
        self.rec3d_out = tf.squeeze(tf.nn.atrous_conv2d(data, weight, 1, padding='SAME'), axis=[2])

    def Sparse(self, texts):
        '''Extract known words into sparse tensors for cost calculations'''
        indices = []
        values = []
        shape = [len(texts), 0]

        for (element, text) in enumerate(texts):

            indexed = [self.characters.index(c) for c in text]

            if len(indexed) > shape[1]:
                shape[1] = len(indexed)

            for (i, index) in enumerate(indexed):
                indices.append([element, i])
                values.append(index)

        return (indices, values, shape)

    def Connectionist_Temporal_Classification(self):
        '''Create the cost and optimiser functions for training, as well as determine the word is when predicting'''
        self.ctc3d_in = tf.transpose(self.rec3d_out, [1, 0, 2])
        self.texts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]), tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))

        # calculation for cost
        self.seq_len = tf.placeholder(tf.int32, [None])
        self.cost = tf.reduce_mean(tf.nn.ctc_loss(inputs=self.ctc3d_in, labels=self.texts, sequence_length=self.seq_len, ctc_merge_repeated=True))
        self.saved_input = tf.placeholder(tf.float32, shape=[Neural_Network.text_len, None, len(self.characters) + 1])
        self.character_cost = tf.nn.ctc_loss(inputs=self.saved_input, labels=self.texts, sequence_length=self.seq_len, ctc_merge_repeated=True)

        # determine word
        self.wordsearch = tf.nn.ctc_greedy_decoder(inputs=self.ctc3d_in, sequence_length=self.seq_len)

    def word_produce(self, ctc_out, batch_size):
        '''Extract texts from CTC'''
        str_codes = [[] for i in range(batch_size)]

        # retrieve Spare Tensor
        output = ctc_out[0][0]

        # map from batch to words
        word_dict = {data: [] for data in range(batch_size)}
        for (index, index_2d) in enumerate(output.indices):
            letter = output.values[index]
            element = index_2d[0]
            str_codes[element].append(letter)

        return [str().join([self.characters[char] for char in word])
                for word in str_codes]

    def batching(self, batch):
        '''Input batches into the neural network'''
        elements = len(batch.images)
        sparse = self.Sparse(batch.texts)
        # decay learning rates, this works for some reason?
        if self.progress < 10:
            rate = 0.01
        elif self.progress < 10000:
            rate = 0.001
        else:
            rate = 0.0001

        evals = [self.optimiser, self.cost]
        
        (_, cost) = self.sess.run(evals, feed_dict={self.batches: batch.images, self.texts: sparse, self.seq_len: [Neural_Network.text_len] * elements, self.rate: rate})
        self.progress += 1
        return cost

    def output(self, rnn_out):
        '''Output the neural network results into CSV files'''
        output_dir = '/Users/naveenkesarwani/Documents/Machine_Learning/SDD Task/outputs/'
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        max_A, max_B, max_C = rnn_out.shape
        for b in range(max_B):
            csv = ''
            for a in range(max_A):
                for c in range(max_C):
                    csv += str(rnn_out[a, b, c]) + ';'
                csv += '\n'
            fn = output_dir + 'rnn_output_' + str(b) + '.csv'
            with open(fn, 'w') as f:
                f.write(csv)

    def test_batch(self, batch):
        '''Test the Neural Network with a batch'''
        elements = len(batch.images)
        rnn_out = self.saved
        eval_req = [self.wordsearch]
        if rnn_out:
            eval_req += [self.ctc3d_in]
        evals = self.sess.run(eval_req, {self.batches: batch.images, self.seq_len: [Neural_Network.text_len] * elements})
        word = evals[0]
        texts = self.word_produce(word, elements)

        if self.saved:
            self.output(evals[1])

        return texts

    def save(self):
        '''Save the Neural Network to a file'''
        self.current += 1
        self.saver.save(self.sess, '/Users/naveenkesarwani/Documents/Machine_Learning/SDD Task/save/latest', global_step=self.current)

