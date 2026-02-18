import math
from collections import Counter

# Calculate entropy of a dataset
def entropy(data, target_attr):
    values = [record[target_attr] for record in data]
    freq = Counter(values)
    total = len(values)
    return sum((-count/total) * math.log2(count/total) for count in freq.values())

# Calculate information gain
def info_gain(data, attr, target_attr):
    total_entropy = entropy(data, target_attr)
    values = set(record[attr] for record in data)
    weighted_entropy = 0
    total = len(data)
    
    for value in values:
        subset = [record for record in data if record[attr] == value]
        weighted_entropy += (len(subset)/total) * entropy(subset, target_attr)
    
    return total_entropy - weighted_entropy

# Choose best attribute
def best_attr(data, attributes, target_attr):
    gains = {attr: info_gain(data, attr, target_attr) for attr in attributes}
    return max(gains, key=gains.get)

# Build decision tree recursively
def id3(data, attributes, target_attr):
    values = [record[target_attr] for record in data]
    # If all examples have same classification
    if values.count(values[0]) == len(values):
        return values[0]
    # If no attributes left
    if not attributes:
        return Counter(values).most_common(1)[0][0]
    
    best = best_attr(data, attributes, target_attr)
    tree = {best: {}}
    
    for value in set(record[best] for record in data):
        subset = [record for record in data if record[best] == value]
        new_attrs = [attr for attr in attributes if attr != best]
        tree[best][value] = id3(subset, new_attrs, target_attr)
    
    return tree

# Example dataset (Play Tennis)
dataset = [
    {"Outlook":"Sunny", "Temp":"Hot", "Humidity":"High", "Wind":"Weak", "Play":"No"},
    {"Outlook":"Sunny", "Temp":"Hot", "Humidity":"High", "Wind":"Strong", "Play":"No"},
    {"Outlook":"Overcast", "Temp":"Hot", "Humidity":"High", "Wind":"Weak", "Play":"Yes"},
    {"Outlook":"Rain", "Temp":"Mild", "Humidity":"High", "Wind":"Weak", "Play":"Yes"},
    {"Outlook":"Rain", "Temp":"Cool", "Humidity":"Normal", "Wind":"Weak", "Play":"Yes"},
    {"Outlook":"Rain", "Temp":"Cool", "Humidity":"Normal", "Wind":"Strong", "Play":"No"},
    {"Outlook":"Overcast", "Temp":"Cool", "Humidity":"Normal", "Wind":"Strong", "Play":"Yes"},
    {"Outlook":"Sunny", "Temp":"Mild", "Humidity":"High", "Wind":"Weak", "Play":"No"},
    {"Outlook":"Sunny", "Temp":"Cool", "Humidity":"Normal", "Wind":"Weak", "Play":"Yes"},
    {"Outlook":"Rain", "Temp":"Mild", "Humidity":"Normal", "Wind":"Weak", "Play":"Yes"},
    {"Outlook":"Sunny", "Temp":"Mild", "Humidity":"Normal", "Wind":"Strong", "Play":"Yes"},
    {"Outlook":"Overcast", "Temp":"Mild", "Humidity":"High", "Wind":"Strong", "Play":"Yes"},
    {"Outlook":"Overcast", "Temp":"Hot", "Humidity":"Normal", "Wind":"Weak", "Play":"Yes"},
    {"Outlook":"Rain", "Temp":"Mild", "Humidity":"High", "Wind":"Strong", "Play":"No"},
]

attributes = ["Outlook", "Temp", "Humidity", "Wind"]
target_attr = "Play"

tree = id3(dataset, attributes, target_attr)
print("Decision Tree:", tree)
