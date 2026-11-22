"""
BSD3513 Introduction to Artificial Intelligence - Lab Report 4
Application 1: Rectified Linear Unit (ReLU) Activation Function Visualizer
Author: Muhammad Haziq Hafizuddin bin MD Masoni
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def relu(x: np.ndarray) -> np.ndarray:
    """Compute the Rectified Linear Unit activation: f(x) = max(0, x)."""
    return np.maximum(0.0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Compute the sub-gradient of ReLU: f'(x) = 1 if x > 0 else 0."""
    return np.where(x > 0, 1.0, 0.0)


def main() -> None:
    st.set_page_config(page_title="ReLU Visualizer | BSD3513", layout="wide")

    st.title("1. Rectified Linear Unit (ReLU) Activation Function")
    st.markdown(
        "Interactive visualization of **ReLU** and its gradient behavior during "
        "neural network forward propagation and backpropagation."
    )

    # Sidebar controls
    st.sidebar.header("Simulation Parameters")
    x_limit = st.sidebar.slider("Input Domain Range (±x)", min_value=2.0, max_value=20.0, value=10.0, step=1.0)
    num_points = st.sidebar.slider("Plot Resolution (Points)", min_value=100, max_value=1000, value=500, step=50)
    probe_x = st.sidebar.number_input("Test Neuron Pre-Activation Input (z)", value=2.5, step=0.5)

    # Generate curve data
    x_vals = np.linspace(-x_limit, x_limit, num_points)
    y_vals = relu(x_vals)
    dy_vals = relu_derivative(x_vals)

    # Point evaluation
    probe_y = float(relu(np.array([probe_x]))[0])
    probe_dy = float(relu_derivative(np.array([probe_x]))[0])
    neuron_state = "ACTIVE (Firing)" if probe_x > 0 else "INACTIVE (Dead Zone)"

    # Formula & Metrics Row
    st.latex(r"f(x) = \max(0, x) \qquad | \qquad f'(x) = \begin{cases} 1, & x > 0 \\ 0, & x \le 0 \end{cases}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Input (z)", f"{probe_x:.2f}")
    col2.metric("Activated Output f(z)", f"{probe_y:.4f}")
    col3.metric("Local Gradient f'(z)", f"{probe_dy:.4f}")
    col4.metric("Neuron State", neuron_state)

    # Dual-Panel Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: Activation Function
    ax1.plot(x_vals, y_vals, color="#2563eb", linewidth=2.5, label="ReLU: max(0, x)")
    ax1.scatter([probe_x], [probe_y], color="#dc2626", zorder=5, s=70, label=f"Probe ({probe_x}, {probe_y:.2f})")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_title("Forward Pass: Activation Output f(x)", fontweight="bold")
    ax1.set_xlabel("Weighted Sum + Bias (x)")
    ax1.set_ylabel("Activation Output f(x)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Panel 2: Derivative
    ax2.plot(x_vals, dy_vals, color="#059669", linewidth=2.5, label="Derivative f'(x)")
    ax2.scatter([probe_x], [probe_dy], color="#dc2626", zorder=5, s=70, label=f"Gradient ({probe_x}, {probe_dy:.2f})")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_ylim(-0.2, 1.3)
    ax2.set_title("Backward Pass: Gradient Flow f'(x)", fontweight="bold")
    ax2.set_xlabel("Weighted Sum + Bias (x)")
    ax2.set_ylabel("Gradient f'(x)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    st.pyplot(fig)

    # Analytical Takeaways
    with st.expander("Analytical Properties & Neural Network Impact", expanded=True):
        st.markdown(
            """
            * **Output Range:** $[0, \\infty)$ — unbounded for positive inputs, preventing upper-bound saturation.
            * **Computational Efficiency:** Requires only a simple threshold comparison (`x > 0`), making it significantly faster than exponential functions (Sigmoid/Tanh).
            * **Vanishing Gradient Mitigation:** Maintains a constant gradient of $1$ for all $x > 0$, enabling effective training of deep architectures.
            * **Limitation (Dying ReLU Problem):** If a large negative bias or high learning rate pushes $x \\le 0$ permanently, $f'(x) = 0$ and the neuron stops updating weights.
            """
        )


if __name__ == "__main__":
    main()