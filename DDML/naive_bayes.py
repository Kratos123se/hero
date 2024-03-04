import csv
import math

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = {}

    def fit(self, X_train, y_train):
        n_samples = len(X_train)
        n_features = len(X_train[0])

        # Compute class probabilities
        self.class_probs = {}
        for label in set(y_train):
            self.class_probs[label] = y_train.count(label) / n_samples

        # Compute feature probabilities
        self.feature_probs = {}
        for label in self.class_probs:
            self.feature_probs[label] = {}
            for i in range(n_features):
                self.feature_probs[label][i] = {}
                for value in set([row[i] for row in X_train]):
                    count = sum(1 for j in range(n_samples) if X_train[j][i] == value and y_train[j] == label)
                    self.feature_probs[label][i][value] = count / y_train.count(label)

    def predict(self, X_test):
        predictions = []
        for sample in X_test:
            max_prob = -1
            max_label = None
            for label in self.class_probs:
                prob = self.class_probs[label]
                for i, value in enumerate(sample):
                    if value in self.feature_probs[label][i]:
                        prob *= self.feature_probs[label][i][value]
                    else:
                        prob *= 1e-6  # Smoothing for unseen values
                if prob > max_prob:
                    max_prob = prob
                    max_label = label
            predictions.append(max_label)
        return predictions

def load_data(filename):
    X = []
    y = []
    with open('/content/train.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            X.append(row[:-1])
            y.append(row[-1])
    return X, y

def accuracy_score(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

# Load training data
X_train, y_train = load_data('training_data.csv')

# Train Naive Bayes classifier
clf = NaiveBayesClassifier()
clf.fit(X_train, y_train)

# Load test data
X_test, y_test = load_data('test_data.csv')

# Predict
y_pred = clf.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)