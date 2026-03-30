import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# UI
st.title("Neural Network Visualizer")
st.write("Learn any function in real time with a neural network.")

# Function input
user_func = st.text_input(
    "Function to Learn (use 'x', numpy as np allowed)",
    "np.sin(x)"
)
st.caption("Examples: np.sin(x), x**2, np.sin(x)+0.5*x, np.exp(-x**2)")

# Learning rate and epochs
learning_rate = st.text_input("Learning Rate", "0.01")
num_epochs = st.text_input("Number of Epochs", "2000")

# Validate inputs
try:
    learning_rate = float(learning_rate)
except:
    st.error("Learning rate must be a number")
    st.stop()

try:
    num_epochs = int(num_epochs)
except:
    st.error("Epochs must be an integer")
    st.stop()

# Activation
activation_choice = st.selectbox(
    "Activation Function",
    ["ReLU", "Tanh", "Sigmoid", "LeakyReLU", "ELU", "GELU"]
)

# Loss
loss_choice = st.selectbox(
    "Loss Function",
    ["MSE", "MAE", "Huber"]
)

# Hidden layers
num_hidden_layers = st.slider("Number of Hidden Layers", 1, 5, 1)
neurons_list = []
fix_all_layers = st.checkbox("Fix all layers same size", True)

for i in range(num_hidden_layers):
    if i == 0:
        n = st.slider(f"Neurons in Layer {i+1}", 1, 100, 10)
    else:
        if fix_all_layers:
            n = neurons_list[0]
        else:
            n = st.slider(f"Neurons in Layer {i+1}", 1, 100, 10)
    neurons_list.append(n)

# Session state for training
if "training" not in st.session_state:
    st.session_state["training"] = False

# Toggle function for button
def toggle_training():
    st.session_state.training = not st.session_state.training

# Single button with callback
button_label = "Stop" if st.session_state.training else "Train"
st.button(button_label, on_click=toggle_training)


# Device, mappings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

activations_dict = {
    "ReLU": nn.ReLU(),
    "Tanh": nn.Tanh(),
    "Sigmoid": nn.Sigmoid(),
    "LeakyReLU": nn.LeakyReLU(),
    "ELU": nn.ELU(),
    "GELU": nn.GELU()
}
activation = activations_dict[activation_choice]

losses_dict = {
    "MSE": nn.MSELoss(),
    "MAE": nn.L1Loss(),
    "Huber": nn.SmoothL1Loss()
}
criterion = losses_dict[loss_choice]

# Data
x = torch.linspace(-4 * torch.pi, 4 * torch.pi, 400).unsqueeze(1).to(device)
safe_dict = {"np": np, "x": x.cpu().numpy().squeeze()}

try:
    y_np = eval(user_func, {"__builtins__": None}, safe_dict)
    y_np = np.array(y_np)
    if y_np.ndim == 0:
        y_np = np.full_like(safe_dict["x"], y_np)
    y_np = y_np.reshape(-1, 1)
    y = torch.tensor(y_np, dtype=torch.float32).to(device)
except Exception as e:
    st.error(f"Invalid function: {e}")
    st.stop()

# Model
class DynamicNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, activation):
        super().__init__()
        layers = []
        last = input_size
        for h in hidden_layers:
            layers.append(nn.Linear(last, h))
            layers.append(activation)
            last = h
        layers.append(nn.Linear(last, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = DynamicNN(1, neurons_list, 1, activation).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Placeholders
plot_placeholder = st.empty()
log_container = st.empty()
log_text = ""
loss_history = []

# Training loop
epoch = 0
while st.session_state.training and epoch < num_epochs:
    model.train()
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if (epoch + 1) % 50 == 0 or epoch == 0:
        # Log
        log_text += f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.6f}\n"
        log_container.text_area("Training Log", log_text, height=200)

        # Plot
        fig, ax = plt.subplots()
        ax.plot(x.cpu().numpy().squeeze(), y.cpu().numpy().squeeze(), label="Ground Truth")
        ax.plot(x.cpu().numpy().squeeze(), y_pred.detach().cpu().numpy().squeeze(), label="Prediction")
        ax.set_title(f"Learning Function - Epoch {epoch+1}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        ax.legend()
        plot_placeholder.pyplot(fig)
        plt.close(fig)

        time.sleep(0.05)

    epoch += 1

# Final loss plot
if loss_history:
    fig2, ax2 = plt.subplots()
    ax2.plot(loss_history, label=f"{loss_choice} Loss")
    ax2.set_title("Loss History")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)
    plt.close(fig2)

st.write("Training complete!" if st.session_state.training else "Training stopped.")