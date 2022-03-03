import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import argparse
from utils import *

tf.config.run_functions_eagerly(True)

class NN(tf.keras.Model):
    def __init__(self, in_size, out_size):
        super(NN, self).__init__()
        
        ######### Your code starts here #########
        # We want to define and initialize the weights & biases of the neural network.
        # - in_size is dim(O)
        # - out_size is dim(A) = 2
        # IMPORTANT: out_size is still 2 in this case, because the action space is 2-dimensional. But your network will output some other size as it is outputing a distribution!
        # HINT: You should use either of the following for weight initialization:
        #         - tf.keras.initializers.GlorotUniform (this is what we tried)
        #         - tf.keras.initializers.GlorotNormal
        #         - tf.keras.initializers.he_uniform or tf.keras.initializers.he_normal
#         model = keras.Sequential(
#     [
#         layers.Dense(2, activation="relu", name="layer1"),
#         layers.Dense(3, activation="relu", name="layer2"),
#         layers.Dense(4, name="layer3"),
#     ]
# )
        
        hidden_size = 15
        # actual_out_size = int(out_size + out_size**2)
        # # activation = 'softmax'
        # initializer = tf.keras.initializers.GlorotUniform()
        # # input_layer = tf.keras.layers.InputLayer(input_shape=in_size)
        # first = tf.keras.layers.Dense(hidden_size, input_shape=(in_size,), activation='sigmoid', kernel_initializer=initializer, use_bias=True, bias_initializer=initializer)#(input_layer)
        # second = tf.keras.layers.Dense(hidden_size, activation='softmax', kernel_initializer=initializer, use_bias=True, bias_initializer=initializer)#(out)
        # # out = tf.keras.layers.Dense(hidden_size, activation='softmax', kernel_initializer=initializer, use_bias=True, bias_initializer=initializer)(out)
        # output_layer = tf.keras.layers.Dense(actual_out_size, kernel_initializer=initializer)#(out) #activation=activation,
        # self.model = tf.keras.Sequential([
        #     # input_layer,
        #     first,
        #     second,
        #     output_layer
        # ])
        # #debugging
        # print(f"in_size: {in_size}")
        # print(f"out_size: {out_size}")
        # # print(f"out_size.shape: {out_size.shape}")
        # #end debugging
        actual_out_size = int(out_size + out_size**2)
        # activation = 'softmax'
        initializer = tf.keras.initializers.GlorotUniform()
        input_layer = tf.keras.layers.Input(shape=in_size)
        out = tf.keras.layers.Dense(hidden_size, activation='sigmoid', kernel_initializer=initializer, use_bias=True, bias_initializer=initializer)(input_layer)
        out = tf.keras.layers.Dense(hidden_size, activation='relu', kernel_initializer=initializer, use_bias=True, bias_initializer=initializer)(out)
        # out = tf.keras.layers.Dense(hidden_size, activation='softmax', kernel_initializer=initializer, use_bias=True, bias_initializer=initializer)(out)
        out = tf.keras.layers.Dense(actual_out_size, kernel_initializer=initializer)(out) #activation=activation,
        #first two columns are the mean vector, the rest is the flattened A matrix such that A@A.T -> covariance matrix
        self.model = tf.keras.Model(inputs=input_layer, outputs=out)
        # print("Model init done")
        
        ########## Your code ends here ##########

    def call(self, x):
        x = tf.cast(x, dtype=tf.float32)
        ######### Your code starts here #########
        # We want to perform a forward-pass of the network. Using the weights and biases, this function should give the network output for x where:
        # x is a (?, |O|) tensor that keeps a batch of observations
        # IMPORTANT: First two columns of the output tensor must correspond to the mean vector!
        # print("Before call")
        output_raw = self.model(x)
        # print("After call")
        # output_matrix = tf.reshape(output_raw[:, 2:6], (-1, 2,2))
        # covar = tf.linalg.matmul(output_matrix, output_matrix, transpose_b=True)
        # covar_vec = tf.reshape(covar, (-1, 4))
        # mean = output_raw[:, 0:2]
        # outputs = tf.concat([mean, covar_vec], -1)
        return output_raw
        
        
        ########## Your code ends here ##########


   
def loss(y_est, y):
    y = tf.cast(y, dtype=tf.float32)
    ######### Your code starts here #########
    # We want to compute the negative log-likelihood loss between y_est and y where
    # - y_est is the output of the network for a batch of observations,
    # - y is the actions the expert took for the corresponding batch of observations
    # At the end your code should return the scalar loss value.
    # HINT: You may find the classes of tensorflow_probability.distributions (imported as tfd) useful.
    #       In particular, you can use MultivariateNormalFullCovariance or MultivariateNormalTriL, but they are not the only way.
    # print("got here")
    batchsize = y_est.shape[0]
    # print("got here 1")
    output_matrix = tf.reshape(y_est[:, 2:6], (batchsize, 2,2))
    # print("got output matrix")
    covar = tf.linalg.matmul(output_matrix, output_matrix, transpose_b=True)
    epsilon = 0.001 * tf.eye(2)
    covar_adj = covar + epsilon
    # print("calculated output matrix")
    # covar_vec = tf.reshape(covar, (-1, 4))
    mean = y_est[:, 0:2]
    dist_est = tfd.MultivariateNormalFullCovariance(loc=mean, covariance_matrix=covar_adj)
    return -tf.reduce_mean(dist_est.log_prob(y))
    # print("got distribution estimate")
    # loglikelihood = tf.math.log(dist_est.prob(y))
    # print("calculated loglikelihood. returning loss")
    # return -(1/batchsize)*tf.reduce_sum(loglikelihood)
#     tfp.distributions.MultivariateNormalFullCovariance(
#     loc=None, covariance_matrix=None, validate_args=False, allow_nan_stats=True,
#     name='MultivariateNormalFullCovariance'
# )
    # outputs = tf.concat([mean, covar_vec], -1)
    
    
    ########## Your code ends here ##########


def nn(data, args):
    """
    Trains a feedforward NN. 
    """
    params = {
        'train_batch_size': 4096*32, #4096*32
    }
    in_size = data['x_train'].shape[-1]
    out_size = data['y_train'].shape[-1]
    nn_model = NN(in_size, out_size)
    if args.restore:
        nn_model.load_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_ILDIST')
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    @tf.function
    def train_step(x, y):
        ######### Your code starts here #########
        # We want to perform a single training step (for one batch):
        # 1. Make a forward pass through the model
        # 2. Calculate the loss for the output of the forward pass
        # 3. Based on the loss calculate the gradient for all weights
        # 4. Run an optimization step on the weights.
        # Helpful Functions: tf.GradientTape(), tf.GradientTape.gradient(), tf.keras.Optimizer.apply_gradients
        with tf.GradientTape() as tape:
            # make forward pass
            # y_est = nn_model(x, training=True)
            y_est = nn_model.call(x)
            # print(y_est.shape)
            vars = nn_model.variables # array of weights and biases
            # print(vars.shape)
            tape.watch(vars)
            # print("tape watched")
            current_loss = loss(y_est,y) # calculate loss
            # print("loss returned")
            grads = tape.gradient(current_loss,vars)
            # print("grads obtained")
        optimizer.apply_gradients(zip(grads, vars)) # one step of GD
       
       
        ########## Your code ends here ##########

        train_loss(current_loss)

    @tf.function
    def train(train_data):
        for x, y in train_data:
            train_step(x, y)


    train_data = tf.data.Dataset.from_tensor_slices((data['x_train'], data['y_train'])).shuffle(100000).batch(params['train_batch_size'])

    for epoch in range(args.epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()

        train(train_data)

        template = 'Epoch {}, Loss: {}'
        print(template.format(epoch + 1, train_loss.result()))
    nn_model.save_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_ILDIST')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--goal', type=str, help="left, straight, right, inner, outer, all", default="all")
    parser.add_argument('--scenario', type=str, help="intersection, circularroad", default="intersection")
    parser.add_argument("--epochs", type=int, help="number of epochs for training", default=1000)
    parser.add_argument("--lr", type=float, help="learning rate for Adam optimizer", default=1e-3)
    parser.add_argument("--restore", action="store_true", default=False)
    args = parser.parse_args()
    
    maybe_makedirs("./policies")
    
    data = load_data(args)

    nn(data, args)


