from __future__ import division, print_function, absolute_import
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
from sklearn.manifold import TSNE
# Import MNIST data
from tensorflow.contrib.timeseries.python.timeseries.saved_model_utils import cold_start_filter

from datasets.kdd99 import input_data

class MyAutoEncoder(object):
    def __init__(self, ninput, layers=[64,32,16]):
        self.tf_sess = tf.Session()
        self.learning_rate = 0.01    # 0.01 this learning rate will be better! Tested
        self.training_epochs = 10
        self.batch_size = 512 
        self.display_step = 1
        # Network Parameters
        self.n_input = ninput
       # hidden layer settings
        self.n_hidden_layers = layers 
        #init layers weights
        nlayers = [self.n_input] + self.n_hidden_layers
       # print(nlayers)
        self.ae_params = {
                'encoder':[
                    {
                        'h':tf.Variable(tf.truncated_normal([nlayers[x-1],nlayers[x]],)),
                        'b':tf.Variable(tf.random_normal([nlayers[x]]))
                    } 
                    for x in range(1,len(nlayers))
                ],
                'decoder':[
                    {
                        'h':tf.Variable(tf.truncated_normal([nlayers[x],nlayers[x-1]],)),
                        'b':tf.Variable(tf.random_normal([nlayers[x-1]]))
                    }
                    for x in range(1,len(nlayers))[::-1]
                ]
            }
        # tf Graph input (only pictures)
        self.X = tf.placeholder("float", [None, self.n_input])
        self.encoder_op = MyAutoEncoder.gen_layers(self.X,self.ae_params['encoder'])
        self.decoder_op = MyAutoEncoder.gen_layers(self.encoder_op,self.ae_params['decoder'])
        # Prediction
        y_pred = self.decoder_op
        y_true = self.X
        self.cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        # tf.initialize_all_variables() no long valid from

        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        self.tf_sess.run(init)

    @staticmethod
    def gen_layers(x,p):
        layers = [x] + [None]*len(p)
        for i in range(len(p)):
            layers[i+1] = tf.nn.sigmoid(tf.add(tf.matmul(layers[i],p[i]['h']),p[i]['b']))
        return layers[-1]

    def fit(self,data):
        cost_history = []
        total_batch = int(data.num_examples/self.batch_size)
        # Training cycle
        for epoch in range(self.training_epochs):
            # Loop over all batches
            it = data.get_epoch()
            for i in range(total_batch):
                batch_xs, batch_ys = it.next_batch(self.batch_size)  # max(x) = 1, min(x) = 0
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = self.tf_sess.run([self.optimizer, self.cost], feed_dict={self.X: batch_xs})
            # Display logs per epoch step
            if epoch % self.display_step == 0:
                print("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(c))
                cost_history.append(c)
        print("Optimization Finished!")
        plt.plot(cost_history)
        plt.xlabel('Epoch')
        plt.ylabel('cost')
        #plt.show()

    def transform(self,data):
        return self.tf_sess.run(self.encoder_op,feed_dict={self.X:data})


if __name__ == "__main__":
    print(datetime.datetime.now(),'start...')
    kdd99 = input_data.read_data_sets('DoS')
    print(datetime.datetime.now(),'data loaded.')
    svm_train_xs,svm_train_ys = kdd99.train.random_select(10000)

    xs,ys = kdd99.train.random_select(500)
    model = MyAutoEncoder([64,32,8])
    print(model.fit(kdd99.train))
    print(datetime.datetime.now(),'autoencoder train finished.')
    encoded_test_xs = model.transform(xs)

    colors = ys
    plt.subplot(2,1,1)
    plt.title('AutoEncoder')
    points = TSNE(random_state=0).fit_transform(encoded_test_xs)
    plt.scatter(points[:,0],points[:,1],c=colors,s=75,alpha=.5)
    plt.show()

