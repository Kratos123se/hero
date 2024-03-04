import numpy as np

class Node:
    def __init__(self, attribute=None, label=None):
        self.attribute = attribute
        self.label = label
        self.children = {}

def entropy(data):
    _, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(data, attribute_index):
    unique_values, counts = np.unique(data[:, attribute_index], return_counts=True)
    total_entropy = entropy(data[:, -1])
    weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * entropy(data[data[:, attribute_index] == value]) for i, value in enumerate(unique_values)])
    return total_entropy - weighted_entropy

def id3(data, attributes):
    labels, label_counts = np.unique(data[:, -1], return_counts=True)
    if len(labels) == 1:
        return Node(label=labels[0])
    if len(attributes) == 0:
        return Node(label=labels[np.argmax(label_counts)])
    information_gains = [information_gain(data, i) for i in attributes]
    best_attribute_index = attributes[np.argmax(information_gains)]
    best_attribute = attributes[np.argmax(information_gains)]
    node = Node(attribute=best_attribute)
    unique_values = np.unique(data[:, best_attribute_index])
    for value in unique_values:
        subset_indices = np.where(data[:, best_attribute_index] == value)[0]
        subset = data[subset_indices]
        if len(subset) == 0:
            labels, label_counts = np.unique(data[:, -1], return_counts=True)
            node.children[value] = Node(label=labels[np.argmax(label_counts)])
        else:
            new_attributes = [attr for attr in attributes if attr != best_attribute]
            node.children[value] = id3(subset, new_attributes)
    return node

def predict(node, sample):
    if node.label is not None:
        return node.label
    attribute_value = sample[node.attribute]
    if attribute_value not in node.children:
        return None
    return predict(node.children[attribute_value], sample)

# Example dataset
# Format: [Outlook, Temperature, Humidity, Windy, PlayTennis]
data = np.array([
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No']
])

attributes = [0, 1, 2, 3]  # Outlook, Temperature, Humidity, Windy

tree = id3(data, attributes)

# Predicting
sample = ['Sunny', 'Cool', 'Normal', 'Strong']  # Example test sample
prediction = predict(tree, sample)
print("Predicted label:", prediction)
