# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

![image](https://github.com/user-attachments/assets/1d0be00b-400c-4ba3-a041-8edfcc79df06)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: HYCINTH D
### Register Number: 212223240055
```
class NeuralNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(1, 8)
    self.fc2 = nn.Linear(8, 10)
    self.fc3 = nn.Linear(10, 1)
    self.relu = nn.ReLU()
    self.history = {'loss':[]}

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)
    return x

ai_brain = NeuralNet()
print(list(ai_brain.parameters()))
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)

def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()  # Clear previous gradients
        loss = criterion(ai_brain(X_train), y_train)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        # Store loss for visualization
        ai_brain.history['loss'].append(loss.item())

        # Print loss every 200 epochs
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


```
## Dataset Information

![DATASET](https://github.com/user-attachments/assets/1af4e5d5-c9a3-4bef-9385-b5a93af3ec57)


## OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot 2025-03-16 071309](https://github.com/user-attachments/assets/5b62eda9-55a0-4c89-a993-c90f01c3dfcf)


### New Sample Data Prediction

![Screenshot 2025-03-16 071611](https://github.com/user-attachments/assets/3bab8474-9858-465b-ad9c-d8f44cb6ff35)


## RESULT

The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
