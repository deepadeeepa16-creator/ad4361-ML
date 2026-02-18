import math
from collections import defaultdict

# Training data
documents = [
    ("I love programming", "positive"),
    ("Python is great", "positive"),
    ("I hate bugs", "negative"),
    ("Debugging is hard", "negative")
]

# Step 1: Training
word_counts = {}
class_counts = defaultdict(int)
vocab = set()

for text, label in documents:
    class_counts[label] += 1
    words = text.lower().split()

    if label not in word_counts:
        word_counts[label] = defaultdict(int)

    for word in words:
        word_counts[label][word] += 1
        vocab.add(word)

total_docs = len(documents)

# Step 2: Prediction function
def predict(text):
    words = text.lower().split()
    scores = {}

    for label in class_counts:
        # Prior Probability
        score = math.log(class_counts[label] / total_docs)

        total_words = sum(word_counts[label].values())

        for word in words:
            # Laplace smoothing
            word_freq = word_counts[label][word] + 1
            score += math.log(word_freq / (total_words + len(vocab)))

        scores[label] = score

    return max(scores, key=scores.get)

# Step 3: Testing
test_text = "I love Python"
print("Document:", test_text)
print("Predicted Class:", predict(test_text))
