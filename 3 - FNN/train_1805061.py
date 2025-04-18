import torchvision.datasets as ds
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle


class AdamOptimizer:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.vw = None
        self.vb = None
        self.sw = None
        self.sb = None
        self.t = 0

    def update(self, w, b, weights_gradient, bias_gradient, learning_rate):
        if self.vw is None:
            self.vw = np.zeros_like(w)
            self.vb = np.zeros_like(b)
            self.sw = np.zeros_like(w)
            self.sb = np.zeros_like(b)

        self.t+=1

        self.vw = self.beta1 * self.vw + (1 - self.beta1) * weights_gradient
        self.vb = self.beta1 * self.vb + (1 - self.beta1) * bias_gradient

        self.sw = self.beta2 * self.sw + (1 - self.beta2) * (weights_gradient ** 2)
        self.sb = self.beta2 * self.sb + (1 - self.beta2) * (bias_gradient ** 2)

        # Bias correction
        vw_corrected = self.vw / (1 - self.beta1 ** self.t)
        vb_corrected = self.vb / (1 - self.beta1 ** self.t)
        sw_corrected = self.sw / (1 - self.beta2 ** self.t)
        sb_corrected = self.sb / (1 - self.beta2 ** self.t)

        w -= learning_rate * vw_corrected / (np.sqrt(sw_corrected) + self.epsilon)
        b -= learning_rate * vb_corrected / (np.sqrt(sb_corrected) + self.epsilon)

        return w, b


class Layer:
      def __init__(self):
            self.input = None
            self.output = None

      def forward(self, input, is_training):
            # TODO: return output
            pass

      def backward(self, output_gradient, learning_rate):
            # TODO: update parameters and return input gradient
            pass


class Dense(Layer):
      def __init__(self, input_size, output_size):
            xavier_stddev = np.sqrt(2 / (input_size + output_size))  # Xavier initialization formula
            self.w = np.random.randn(output_size, input_size) * xavier_stddev
            self.b = np.zeros((output_size, 1))
            self.adam = AdamOptimizer()
      
      def forward(self, input, is_training=False):
            self.input = input
            self.output = np.dot(self.w, self.input) + self.b
            return self.output
      
      def backward(self, output_gradient, learning_rate):
            m = output_gradient.shape[1]
            weights_gradient = np.dot(output_gradient, self.input.T) / m
            bias_gradient = np.mean(output_gradient, axis=1, keepdims=True)
            input_gradient = np.dot(self.w.T, output_gradient) / m
            self.w , self.b = self.adam.update(self.w, self.b, weights_gradient, bias_gradient, learning_rate)
            return input_gradient


class ReLU(Layer):
      def __init__(self):
            self.mask = None
      
      def forward(self, input, is_training=False):
            self.input = input
            self.mask = self.input > 0
            self.output = self.input * self.mask
            return self.output
      
      def backward(self, output_gradient, learning_rate):
            input_gradient = output_gradient * self.mask
            return input_gradient


class Dropout(Layer):
    def __init__(self, dropout_prob):
        self.dropout_prob = dropout_prob
        self.mask = None
    
    def forward(self, input, is_training=False):
        self.input = input
        if is_training:
            self.mask = np.random.rand(*self.input.shape) > self.dropout_prob
            self.output = self.input * self.mask / (1 - self.dropout_prob)
        else:
            self.output = self.input
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        input_gradient = output_gradient * self.mask / (1 - self.dropout_prob)
        return input_gradient


class Softmax(Layer):
    def __init__(self):
         pass
    
    def forward(self, input, is_training=False):
        self.input = input
        exp_input = np.exp(self.input - np.max(self.input, axis=0, keepdims=True))
        self.output = exp_input / np.sum(exp_input, axis=0, keepdims=True)
        return self.output

    def backward(self, y_true, learning_rate):
        input_gradient =  self.output - y_true
        return input_gradient


def categorical_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predicted values
    return -np.sum(y_true * np.log(y_pred))


def linear_lr_scheduler(initial_lr, final_lr, num_epochs, epoch):
    return initial_lr - ((initial_lr - final_lr) * epoch / num_epochs)


def predict(network, input, is_training=False):
    output = input
    for layer in network:
        output = layer.forward(output, is_training)
    return output


def train(network, x_train, y_train, epochs=50, initial_lr=0.005, final_lr = 5e-6, batch_size=1024):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_f1_scores = []
    val_f1_scores = []
    for e in range(epochs):
        total_error = 0
        batches_per_epoch = int(np.ceil(len(x_train) / batch_size))
        pbar = tqdm(total=batches_per_epoch, desc=f'Epoch {e + 1}/{epochs}', unit='batch')

        # Shuffle the training data before each epoch
        indices = np.random.permutation(len(x_train))
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]

        for i in range(0, len(x_train), batch_size):
            x_batch = x_train_shuffled[i:i + batch_size].T
            y_batch = y_train_shuffled[i:i + batch_size].T

            # Forward pass for the batch
            output = predict(network, x_batch, is_training=True)
            
            # Calculate error for the batch
            loss = categorical_cross_entropy(y_batch, output)
            total_error += loss

            # Compute feedback for the batch
            feedback = y_batch
            for layer in reversed(network):
                feedback = layer.backward(feedback, linear_lr_scheduler(initial_lr, final_lr, epochs, e))
            
            pbar.update(1)  # Update progress bar by batch count
            pbar.set_postfix({'loss': loss/(min(len(x_train), i+batch_size)-i)})  # Update displayed loss

        pbar.close()
        average_loss = total_error / len(x_train)
        print(f'Epoch {e + 1}/{epochs} - Average Loss: {average_loss:.4f}')

        train_loss, train_accuracy, train_f1_score = get_metrics(network, x_train, y_train)
        val_loss, val_accuracy, val_f1_score = get_metrics(network, x_val, y_val)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        train_f1_scores.append(train_f1_score)
        val_f1_scores.append(val_f1_score)
    
    return train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores


def get_metrics(network, x_test, y_test):
    output_list = predict(network, x_test.T)

    loss = categorical_cross_entropy(y_test.T, output_list)/y_test.shape[0]

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(output_list, axis=0).flatten()

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    return loss, accuracy, f1


def get_confusion_matrix(network, x_test, y_test):
    output_list = predict(network, x_test.T)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(output_list, axis=0).flatten()
    return confusion_matrix(y_pred=y_pred, y_true=y_true)


def read_data():
    train_validation_dataset = ds.EMNIST(root='./data', split='letters',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)

    independent_test_dataset = ds.EMNIST(root='./data', split='letters',
                                train=False,
                                transform=transforms.ToTensor())

    train_dataset, validation_dataset = train_test_split(train_validation_dataset, test_size=0.15, random_state=42)
    return train_dataset, validation_dataset, independent_test_dataset


def prepare_data(dataset):
    inputs = []
    outputs = []
    for data in dataset:
        image, label = data
        # Flatten the image tensor and convert to NumPy array
        inputs.append(image.reshape(28*28))
        # Convert label to one-hot encoded format
        output = np.zeros(26) 
        output[label-1] = 1
        outputs.append(output)
    return np.array(inputs), np.array(outputs)


def plot_data(x, y):
    _, axes = plt.subplots(1, 1, figsize=(5, 4))
    image = x.reshape(28, 28)
    label = np.argmax(y)
    axes.imshow(image, cmap='gray')
    axes.set_title(f"Label: {label}")
    axes.axis('off')
    plt.tight_layout()
    plt.show()


def plot_metrics(train_data, val_data, y_axis_label):
    epochs = len(train_data)

    plt.figure(figsize=(8, 6))
    epochs_range = range(1, epochs + 1)

    plt.plot(epochs_range, train_data, label='Training')
    plt.plot(epochs_range, val_data, label='Validation')

    plt.title(f'Training and Validation {y_axis_label} over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(y_axis_label)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{y_axis_label}.jpg', dpi=300, bbox_inches='tight')


def plot_confusion_matrix(confusion_matrix):
    plt.figure(figsize=(12, 12))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig('Confusion Matrix.jpg', dpi=300, bbox_inches='tight')


def combine_plots():
    images = ['Loss.jpg', 'Accuracy.jpg', 'F1 Score.jpg']
    combined_image = plt.figure(figsize=(12, 4))

    for i, image in enumerate(images, start=1):
        img = plt.imread(image)
        combined_image.add_subplot(1, 3, i)
        plt.imshow(img)
        plt.axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig('Combined.jpg', dpi=300, bbox_inches='tight')


def save_model(network):
    for layer in network:
        layer.input = None
        layer.output = None
        if  isinstance(layer, Dropout) or isinstance(layer, ReLU):
            layer.mask = None
        if isinstance(layer, Dense):
            layer.adam = None

    # Save the model using pickle
    with open('model_1805061.pickle', 'wb') as file:
        pickle.dump(network, file)
             

if __name__ == "__main__":
    np.random.seed(42)

    train_dataset, validation_dataset, _ = read_data()
    x_train, y_train = prepare_data(train_dataset)
    x_val, y_val = prepare_data(validation_dataset)

    network = [
        Dense(28*28, 1024),
        ReLU(),
        Dropout(0.3),
        Dense(1024, 26),
        Softmax()
    ]

    # Train the network
    train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores = train(network, x_train, y_train, epochs=15, initial_lr=0.005, batch_size=1024)
    
    # Plot graphs
    plot_metrics(train_losses, val_losses, 'Loss')
    plot_metrics(train_accuracies, val_accuracies, 'Accuracy')
    plot_metrics(train_f1_scores, val_f1_scores, 'F1 Score')
    combine_plots()

    plot_confusion_matrix(get_confusion_matrix(network, x_val, y_val))

    save_model(network)

