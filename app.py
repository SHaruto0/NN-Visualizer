import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from model import DynamicNN
from configs import activations_dict, losses_dict
from inputs import input_activation, input_epochs, input_func, input_loss, input_lr, input_neurons, input_train
from utils import init_data, plot_curves, plot_loss

# UI
st.title("Neural Network Visualizer")
st.write("Learn any function in real time with a neural network.")

# Inputs
user_func = input_func()
learning_rate = input_lr()
num_epochs = input_epochs()

# Activation
activation_choice = input_activation()
# Loss
loss_choice = input_loss()

# Hidden layers
neurons_list = input_neurons()

# Session state for training
input_train()

# Device, mappings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

activation = activations_dict[activation_choice]
criterion = losses_dict[loss_choice]

if st.session_state.training:
    # Data
    x, y = init_data(user_func, device)

    model = DynamicNN(1, neurons_list, 1, activation).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Placeholders
    plot_placeholder = st.empty()
    loss_plot_placeholder = st.empty()
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
            fig = plot_curves(x, y, y_pred, epoch)
            plot_placeholder.pyplot(fig)
            plt.close(fig)
            fig = plot_loss(loss_history, loss_choice)
            loss_plot_placeholder.pyplot(fig)
            plt.close(fig)

            time.sleep(0.05)

        epoch += 1

    # Final loss plot
    if loss_history:
        fig = plot_loss(loss_history, loss_choice)
        loss_plot_placeholder.pyplot(fig)
        plt.close(fig)

    st.write("Training complete!" if st.session_state.training else "Training stopped.")