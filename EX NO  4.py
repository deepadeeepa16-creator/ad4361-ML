import math

# Sample dataset
X = [[1.0, 2.1],
     [1.5, 1.8],
     [5.0, 8.0],
     [6.0, 9.0],
     [1.2, 0.8],
     [7.0, 10.0]]

y = [0, 0, 1, 1, 0, 1]


class NaiveBayes:
    def fit(self, X, y):
        self.classes = list(set(y))
        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:
            X_c = [X[i] for i in range(len(X)) if y[i] == c]
            self.priors[c] = len(X_c) / len(X)

            # Calculate mean
            self.mean[c] = []
            for col in range(len(X[0])):
                mean_val = sum(row[col] for row in X_c) / len(X_c)
                self.mean[c].append(mean_val)

            # Calculate variance
            self.var[c] = []
            for col in range(len(X[0])):
                mean_val = self.mean[c][col]
                variance = sum((row[col] - mean_val) ** 2 for row in X_c) / len(X_c)
                self.var[c].append(variance)

    def gaussian_pdf(self, mean, var, x):
        if var == 0:
            var = 1e-6  # prevent division by zero
        exponent = math.exp(-((x - mean) ** 2) / (2 * var))
        return (1 / math.sqrt(2 * math.pi * var)) * exponent

    def predict(self, X):
        predictions = []

        for row in X:
            posteriors = {}

            for c in self.classes:
                prior = math.log(self.priors[c])
                conditional = 0

                for i in range(len(row)):
                    mean = self.mean[c][i]
                    var = self.var[c][i]
                    conditional += math.log(self.gaussian_pdf(mean, var, row[i]))

                posteriors[c] = prior + conditional

            predictions.append(max(posteriors, key=posteriors.get))

        return predictions


# Train model
model = NaiveBayes()
model.fit(X, y)

# Test data
X_test = [[2.0, 3.0], [8.0, 9.0]]

predictions = model.predict(X_test)

print("Predictions:", predictions)
