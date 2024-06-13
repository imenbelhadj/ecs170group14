import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# Define the neural network architecture for the AI agent
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()  # Initialize the superclass to inherit from nn.Module
        self.linear1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.linear2 = nn.Linear(hidden_size, output_size)  # Second fully connected layer

    def forward(self, x):
        x = F.relu(self.linear1(x))  # Activation function for hidden layer
        x = self.linear2(x)  # Output layer without activation (common in Q-learning)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'  # Path to save the model
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)  # Create directory if it does not exist
        file_name = os.path.join(model_folder_path, file_name)  # Full path to file
        torch.save(self.state_dict(), file_name)  # Save the model state

# Define the trainer class for the AI agent
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr  # Learning rate
        self.gamma = gamma  # Discount factor for future rewards
        self.model = model  # Neural network model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)  # Adam optimizer
        self.criterion = nn.MSELoss()  # Mean squared error loss

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)  # Convert state to tensor
        next_state = torch.tensor(next_state, dtype=torch.float)  # Convert next state to tensor
        action = torch.tensor(action, dtype=torch.long)  # Convert action to tensor
        reward = torch.tensor(reward, dtype=torch.float)  # Convert reward to tensor

        if len(state.shape) == 1:  # Convert 1D state to 2D if necessary
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.model(state)  # Predicted Q-values from the model for the current state
        target = pred.clone()  # Duplicate predictions to use as target for loss calculation

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))  # Update rule for Q-learning
            target[idx][torch.argmax(action[idx]).item()] = Q_new  # Set the target for loss calculation

        self.optimizer.zero_grad()  # Zero the gradients to prevent accumulation
        loss = self.criterion(target, pred)  # Compute the loss
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Update model weights
