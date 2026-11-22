"""
BSD3513 Introduction to Artificial Intelligence - Lab Report 4
Application 2: Logistic Sigmoid Activation Function Visualizer
Author: Muhammad Haziq Hafizuddin bin MD Masoni
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute numerically stable Sigmoid activation: f(x) = 1 / (1 + exp(-x))."""
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Compute Sigmoid derivative: f'(x) = f(x) * (1 - f(x))."""
    s = sigmoid(x)
    return s * (1.0 - s)


def main() -> None:
    st.set_page_config(page_title="Sigmoid Visualizer | BSD3513", layout="wide")

    st.title("2. Logistic Sigmoid Activation Function")
    st.markdown(
        "Interactive visualization of the **Sigmoid** activation function, probability "
        "mapping, and gradient saturation behavior."
    )

    # Sidebar controls
    st.sidebar.header("Simulation Parameters")
    x_limit = st.sidebar.slider("Input Domain Range (±x)", min_value=4.0, max_value=20.0, value=10.0, step=1.0)
    num_points = st.sidebar.slider("Plot Resolution (Points)", min_value=100, max_value=1000, value=500, step=50)
    probe_x = st.sidebar.number_input("Test Neuron Pre-Activation Input (z)", value=0.0, step=0.5)

    # Generate curve data
    x_vals = np.linspace(-x_limit, x_limit, num_points)
    y_vals = sigmoid(x_vals)
    dy_vals = sigmoid_derivative(x_vals)

    # Point evaluation
    probe_y = float(sigmoid(np.array([probe_x]))[0])
    probe_dy = float(sigmoid_derivative(np.array([probe_x]))[0])
    regime = "Linear / Active Zone" if abs(probe_x) <= 3.0 else "Saturation Zone (Vanishing Gradient)"

    # Formula & Metrics Row
    st.latex(r"\sigma(x) = \frac{1}{1 + e^{-x}} \qquad | \qquad \sigma'(x) = \sigma(x)\bigl(1 - \sigma(x)\bigr)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Input (z)", f"{probe_x:.2f}")
    col2.metric("Probability Output σ(z)", f"{probe_y:.4f}")
    col3.metric("Local Gradient σ'(z)", f"{probe_dy:.4f}")
    col4.metric("Operating Regime", regime)

    # Dual-Panel Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: Activation Function
    ax1.plot(x_vals, y_vals, color="#7c3aed", linewidth=2.5, label="Sigmoid: 1 / (1 + e⁻ˣ)")
    ax1.scatter([probe_x], [probe_y], color="#dc2626", zorder=5, s=70, label=f"Probe ({probe_x}, {probe_y:.3f})")
    ax1.axhline(0.5, color="orange", linestyle=":", linewidth=1.0, label="Decision Boundary (0.5)")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.axhline(1, color="gray", linestyle="--", linewidth=0.8)
    ax1.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_ylim(-0.08, 1.08)
    ax1.set_title("Forward Pass: S-Curve Probability Mapping σ(x)", fontweight="bold")
    ax1.set_xlabel("Weighted Sum + Bias (x)")
    ax1.set_ylabel("Activation Output σ(x)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Panel 2: Derivative
    ax2.plot(x_vals, dy_vals, color="#059669", linewidth=2.5, label="Derivative σ'(x)")
    ax2.scatter([probe_x], [probe_dy], color="#dc2626", zorder=5, s=70, label=f"Gradient ({probe_x}, {probe_dy:.3f})")
    ax2.axhline(0.25, color="orange", linestyle=":", linewidth=1.0, label="Max Gradient (0.25 at x=0)")
    ax2.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_ylim(-0.02, 0.30)
    ax2.set_title("Backward Pass: Gradient Bell Curve σ'(x)", fontweight="bold")
    ax2.set_xlabel("Weighted Sum + Bias (x)")
    ax2.set_ylabel("Gradient σ'(x)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    st.pyplot(fig)

    # Analytical Takeaways
    with st.expander("Analytical Properties & Neural Network Impact", expanded=True):
        st.markdown(
            """
            * **Output Range:** $(0, 1)$ — squashes any real-valued input into a bounded probability, making it ideal for the **output layer of binary classifiers**.
            * **Smooth & Differentiable:** Provides continuous gradients everywhere with no sharp corners.
            * **Limitation 1 (Vanishing Gradients):** The maximum derivative is $0.25$ at $x = 0$, and for $|x| > 5$, $\\sigma'(x) \\approx 0$. Multiplying small gradients across many hidden layers causes early-layer weights to stop learning.
            * **Limitation 2 (Non-Zero-Centered):** Because outputs are always positive $(0, 1)$, gradients on weights all share the same sign during backpropagation, causing zig-zagging optimization updates.
            """
        )


if __name__ == "__main__":
    main()