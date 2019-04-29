import numpy as np

from activation_functions import ReLU, grad_ReLU, softmax, grad_softmax


class TwoLayerNet:
    def __init__(self, input_size, hidden_size ,output_size, dropout_p, std=1e-4):
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.dropout_p = dropout_p

    def loss(self, X, y=None, reg=0.0):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        """
            Get scores and perform forward pass
        """
        scores = None

        # Layer 1
        # (N, D)x(D, H) + (1, H) = (N, H)
        scores = np.dot(X, W1)+b1
        relu_activations = ReLU(scores)

        # Dropout after layer 1
        # Kill whole neurons
        drop1 = np.random.randn(scores.shape)
        relu_activations[:, drop1 <= self.dropout_p] = 0

        # Layer 2
        scores = relu_activations.dot(W2)+b2

        if y is None:
            return scores

        softmax_scores = softmax(scores)
        scores = -np.log(softmax_scores[range(N), y])
        loss = np.mean(scores)
        loss += reg*(np.sum(W1*W1)+np.sum(W2*W2)+np.sum(b1*b1)+np.sum(b2*b2))

        """
            Calculating gradients via backprop
        """

        grads = {}

        # Differentiate loss wrt scores for each class
        dsoft = softmax_scores.copy()
        dsoft[range(N), y] -= 1
        dsoft /= N

        dW2 = relu_activations.T.dot(dsoft)
        dW2 += 2*reg*W2
        grads['dW2'] = dW2

        db2 = dsoft*1
        grads['db2'] = np.sum(db2, axis=0)

        dx2 = np.dot(dsoft, W2.T)
        relu_ones = (relu_activations >= 0)*1
        # Only allow those gradients to flow back whose activations were positive
        drelu = dx2*relu_ones

        dW1 = X.T.dot(drelu)
        dW1 += 2*reg*W1
        grads['dW1'] = dW1

        db1 = drelu*1
        grads['db1'] = np.sum(db1, axis=0)

        return loss, grads

    def predict(self, X):
        # Higher score => higher softmaxed probability
        # So picking highest score is equivalent to picking highest softmax
        # probability
        scores = self.loss(X)
        y_pred = np.argmax(scores, axis=1)

        return y_pred

    def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            
            rand_perm = np.random.permutation(num_train)[:batch_size]
            X_batch = X[rand_perm, :]
            y_batch = y[rand_perm]

            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            self.params['W1'] -= learning_rate*grads['dW1']
            self.params['W2'] -= learning_rate*grads['dW2']
            self.params['b1'] -= learning_rate*grads['db1']
            self.params['b2'] -= learning_rate*grads['db2']
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

