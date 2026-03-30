import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

def init_data(user_func, device):
    x = torch.linspace(-4 * torch.pi, 4 * torch.pi, 400).unsqueeze(1).to(device)
    safe_dict = {"np": np, "x": x.cpu().numpy().squeeze()}

    try:
        y_np = eval(user_func, {"__builtins__": None}, safe_dict)
        y_np = np.array(y_np)
        if y_np.ndim == 0:
            y_np = np.full_like(safe_dict["x"], y_np)
        y_np = y_np.reshape(-1, 1)
        y = torch.tensor(y_np, dtype=torch.float32).to(device)
        return x, y
    except Exception as e:
        st.error(f"Invalid function: {e}")
        st.stop()

def plot_curves(x, y, y_pred, epoch):
    fig, ax = plt.subplots()
    ax.plot(x.cpu().numpy().squeeze(), y.cpu().numpy().squeeze(), label="Ground Truth")
    ax.plot(x.cpu().numpy().squeeze(), y_pred.detach().cpu().numpy().squeeze(), label="Prediction")
    ax.set_title(f"Learning Function - Epoch {epoch+1}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.legend()

    return fig

def plot_loss(loss_history, loss_choice):
    fig, ax = plt.subplots()
    ax.plot(loss_history, label=f"{loss_choice} Loss")
    ax.set_title("Loss History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()

    return fig
