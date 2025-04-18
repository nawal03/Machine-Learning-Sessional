import pickle
from train_1805061 import AdamOptimizer, Dense, ReLU, Dropout, Softmax, predict, get_metrics, read_data, prepare_data

# Load the saved model
with open('model_1805061.pickle', 'rb') as file:
    network = pickle.load(file)

_, _,independent_test_dataset = read_data()

x_test, y_test = prepare_data(independent_test_dataset)

loss, accuracy, f1 = get_metrics(network, x_test, y_test)

print(f'Loss: {round(loss,4)}')
print(f'Accuracy: {round(accuracy,4)}')
print(f'F1 Score: {round(f1,4)}')
