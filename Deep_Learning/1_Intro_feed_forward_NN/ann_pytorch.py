import torch
import torch.nn as nn
import torch.optim as optim
https://saturncloud.io/blog/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way/
# Define a simple feedforward neural network
class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# Hyperparameters
input_size = 10
hidden_size = 20
output_size = 1
learning_rate = 0.001
num_epochs = 1000

# Create the model, loss function, and optimizer
model = BinaryClassifier(input_size, hidden_size, output_size)
criterion = nn.BCELoss()  # Binary Cross-Entropy loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Generate some example data
num_samples = 1000
X = torch.randn(num_samples, input_size)
y = torch.randint(2, (num_samples, 1), dtype=torch.float32)
print(X.shape)
# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X)
    plt.plot(outputs)
    
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
test_samples = 10
test_data = torch.randn(test_samples, input_size)
with torch.no_grad():
    predictions = model(test_data)
    predicted_classes = (predictions >= 0.5).float()

print("Test Data:")
print(test_data)
print("Predictions:")
print(predicted_classes)
