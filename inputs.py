import streamlit as st

def input_func():
    # Function input
    user_func = st.text_input(
        "Function to Learn (use 'x', numpy as np allowed)",
        "np.sin(x)"
    )
    st.caption("Examples: np.sin(x), x**2, np.sin(x)+0.5*x, np.exp(-x**2)")
    return user_func

def input_lr():
    # Learning rate
    learning_rate = st.text_input("Learning Rate", "0.01")
    try:
        learning_rate = float(learning_rate)
    except:
        st.error("Learning rate must be a number")
        st.stop()
    return learning_rate

def input_epochs():
    # Epochs
    num_epochs = st.text_input("Number of Epochs", "2000")
    try:
        num_epochs = int(num_epochs)
    except:
        st.error("Epochs must be an integer")
        st.stop()
    return num_epochs

def input_activation():
    # Activation
    activation_choice = st.selectbox(
        "Activation Function",
        ["ReLU", "Tanh", "Sigmoid", "LeakyReLU", "ELU", "GELU"]
    )
    return activation_choice

def input_loss():
    # Loss
    loss_choice = st.selectbox(
        "Loss Function",
        ["MSE", "MAE", "Huber"]
    )
    return loss_choice

def input_neurons():
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

    return neurons_list

def input_train():
    # Session state for training
    if "training" not in st.session_state:
        st.session_state["training"] = False

    # Toggle function for button
    def toggle_training():
        st.session_state.training = not st.session_state.training

    # Single button with callback
    button_label = "Stop" if st.session_state.training else "Train"
    st.button(button_label, on_click=toggle_training)

def input_all():
    user_func = input_func()
    learning_rate = input_lr()
    num_epochs = input_epochs()
    activation_choice = input_activation()
    loss_choice = input_loss()
    neurons_list = input_neurons()
    input_train()

    return user_func, learning_rate, num_epochs, activation_choice, loss_choice, neurons_list