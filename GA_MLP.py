import numpy as np
import matplotlib.pyplot as plt
import random

# Load and preprocess dataset
data = []
with open("GA_dataset.txt", "r") as f:
    for line in f:
        values = line.strip().split(",")
        label = 1 if values[1] == 'M' else 0
        features = list(map(float, values[2:]))
        data.append([label] + features)

data = np.array(data)
X = data[:, 1:]  # Features only
y = data[:, 0].astype(int)  # Labels (0 or 1 for binary classification)

# Feature scaling (manual standardization)
means = np.mean(X, axis=0)
stds = np.std(X, axis=0)
X = (X - means) / stds

# Genetic Algorithm Parameters
population_size = 500
num_generations = 350
mutation_rate = 0.1
crossover_rate = 0.1
k_folds = 10

# Set random seed for reproducibility
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)

# Helper function to create a single MLP individual
def create_individual(input_size, hidden_layers):
    individual = []
    for i in range(len(hidden_layers)):
        layer_input_size = input_size if i == 0 else hidden_layers[i - 1]
        weights = np.random.randn(layer_input_size, hidden_layers[i])
        bias = np.random.randn(hidden_layers[i])
        individual.extend(weights.flatten())
        individual.extend(bias)
    weights_output = np.random.randn(hidden_layers[-1], 1)
    bias_output = np.random.randn(1)
    individual.extend(weights_output.flatten())
    individual.extend(bias_output)
    return np.array(individual)

def initialize_population(input_size, hidden_layers):
    return [create_individual(input_size, hidden_layers) for _ in range(population_size)]

def forward_pass(individual, x, input_size, hidden_layers):
    idx = 0
    for layer_size in hidden_layers:
        weights = individual[idx:idx + input_size * layer_size].reshape(input_size, layer_size)
        idx += input_size * layer_size
        bias = individual[idx:idx + layer_size]
        idx += layer_size
        x = np.tanh(np.dot(x, weights) + bias)
        input_size = layer_size
    weights_output = individual[idx:idx + hidden_layers[-1] * 1].reshape(hidden_layers[-1], 1)
    bias_output = individual[idx + hidden_layers[-1] * 1:]
    output_layer = np.dot(x, weights_output) + bias_output
    return 1 / (1 + np.exp(-output_layer))

def fitness(individual, X_train, y_train, input_size, hidden_layers):
    predictions = np.array([forward_pass(individual, x, input_size, hidden_layers) for x in X_train])
    predictions = (predictions > 0.5).astype(int)
    accuracy = np.mean(predictions.flatten() == y_train)
    return accuracy

def cross_validate(population, input_size, hidden_layers):
    fold_size = len(X) // k_folds
    accuracies = []
    all_predictions = []
    all_labels = []

    for fold in range(k_folds):
        X_train = np.concatenate([X[:fold * fold_size], X[(fold + 1) * fold_size:]], axis=0)
        y_train = np.concatenate([y[:fold * fold_size], y[(fold + 1) * fold_size:]], axis=0)
        X_test = X[fold * fold_size:(fold + 1) * fold_size]
        y_test = y[fold * fold_size:(fold + 1) * fold_size]
        
        # Evolve the population using GA and get the best individual for this fold
        best_individual = max(population, key=lambda ind: fitness(ind, X_train, y_train, input_size, hidden_layers))
        
        # Calculate predictions on the test set
        test_predictions = (np.array([forward_pass(best_individual, x, input_size, hidden_layers) for x in X_test]) > 0.5).astype(int).flatten()
        all_predictions.extend(test_predictions)
        all_labels.extend(y_test)

        # Calculate fitness on the test set
        test_accuracy = fitness(best_individual, X_test, y_test, input_size, hidden_layers)
        accuracies.append(test_accuracy)
    
    return np.mean(accuracies), np.array(all_labels), np.array(all_predictions)

# Confusion matrix function
def calculate_confusion_matrix(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    return cm

def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Benign (B)', 'Malignant (M)'])
    plt.yticks(tick_marks, ['Benign (B)', 'Malignant (M)'])
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# Experiment with different hidden layer structures
hidden_layer_options = [[10]]
results = {}
trial_accuracies = []

# Conduct multiple trials and average the results to reduce variance
num_trials = 3
for hidden_layers in hidden_layer_options:
    print(f"Testing hidden layers: {hidden_layers}")
    accuracies = []
    for _ in range(num_trials):
        population = initialize_population(X.shape[1], hidden_layers)
        avg_accuracy, y_true, y_pred = cross_validate(population, X.shape[1], hidden_layers)
        accuracies.append(avg_accuracy)
    avg_accuracy = np.mean(accuracies)
    trial_accuracies.append(accuracies)  # Store for plotting
    results[tuple(hidden_layers)] = avg_accuracy
    print(f"Average cross-validation accuracy over {num_trials} trials: {avg_accuracy:.2f}")

    # Calculate and plot confusion matrix using only test data
    cm = calculate_confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm)

# Plot accuracy over trials
plt.figure()
for idx, hidden_layers in enumerate(hidden_layer_options):
    plt.plot(range(1, num_trials + 1), trial_accuracies[idx], label=f'Hidden layers: {hidden_layers}')
plt.xlabel('Trial')
plt.ylabel('Accuracy')
plt.title('Cross-validation Accuracy per Trial')
plt.legend()
plt.show()
