"""
BSD3513 Introduction to Artificial Intelligence - Lab Report 4
Application 3: Hyperbolic Tangent (Tanh) Activation Function Visualizer
Author: Muhammad Haziq Hafizuddin bin MD Masoni
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def tanh(x: np.ndarray) -> np.ndarray:
    """Compute Hyperbolic Tangent activation: f(x) = (e^x - e^-x) / (e^x + e^-x)."""
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Compute Tanh derivative: f'(x) = 1 - tanh^2(x)."""
    t = np.tanh(x)
    return 1.0 - (t ** 2)


def main() -> None:
    st.set_page_config(page_title="Tanh Visualizer | BSD3513", layout="wide")

    st.title("3. Hyperbolic Tangent (Tanh) Activation Function")
    st.markdown(
        "Interactive visualization of the zero-centered **Tanh** activation function "
        "and its steeper gradient profile compared to Sigmoid."
    )

    # Sidebar controls
    st.sidebar.header("Simulation Parameters")
    x_limit = st.sidebar.slider("Input Domain Range (±x)", min_value=3.0, max_value=20.0, value=8.0, step=1.0)
    num_points = st.sidebar.slider("Plot Resolution (Points)", min_value=100, max_value=1000, value=500, step=50)
    probe_x = st.sidebar.number_input("Test Neuron Pre-Activation Input (z)", value=0.0, step=0.5)

    # Generate curve data
    x_vals = np.linspace(-x_limit, x_limit, num_points)
    y_vals = tanh(x_vals)
    dy_vals = tanh_derivative(x_vals)

    # Point evaluation
    probe_y = float(tanh(np.array([probe_x]))[0])
    probe_dy = float(tanh_derivative(np.array([probe_x]))[0])
    regime = "High-Sensitivity Zone" if abs(probe_x) <= 2.5 else "Saturation Zone (±1 Asymptote)"

    # Formula & Metrics Row
    st.latex(r"\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \qquad | \qquad \tanh'(x) = 1 - \tanh^2(x)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Input (z)", f"{probe_x:.2f}")
    col2.metric("Activated Output tanh(z)", f"{probe_y:.4f}")
    col3.metric("Local Gradient tanh'(z)", f"{probe_dy:.4f}")
    col4.metric("Operating Regime", regime)

    # Dual-Panel Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: Activation Function
    ax1.plot(x_vals, y_vals, color="#ea580c", linewidth=2.5, label="Tanh(x)")
    ax1.scatter([probe_x], [probe_y], color="#dc2626", zorder=5, s=70, label=f"Probe ({probe_x}, {probe_y:.3f})")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.axhline(1, color="gray", linestyle=":", linewidth=0.8)
    ax1.axhline(-1, color="gray", linestyle=":", linewidth=0.8)
    ax1.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_ylim(-1.15, 1.15)
    ax1.set_title("Forward Pass: Zero-Centered S-Curve tanh(x)", fontweight="bold")
    ax1.set_xlabel("Weighted Sum + Bias (x)")
    ax1.set_ylabel("Activation Output tanh(x)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Panel 2: Derivative
    ax2.plot(x_vals, dy_vals, color="#059669", linewidth=2.5, label="Derivative tanh'(x)")
    ax2.scatter([probe_x], [probe_dy], color="#dc2626", zorder=5, s=70, label=f"Gradient ({probe_x}, {probe_dy:.3f})")
    ax2.axhline(1.0, color="orange", linestyle=":", linewidth=1.0, label="Max Gradient (1.0 at x=0)")
    ax2.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_ylim(-0.05, 1.15)
    ax2.set_title("Backward Pass: Steeper Gradient Profile tanh'(x)", fontweight="bold")
    ax2.set_xlabel("Weighted Sum + Bias (x)")
    ax2.set_ylabel("Gradient tanh'(x)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    st.pyplot(fig)

    # Analytical Takeaways
    with st.expander("Analytical Properties & Neural Network Impact", expanded=True):
        st.markdown(
            """
            * **Output Range:** $(-1, 1)$ — symmetric around the origin ($\\tanh(0) = 0$).
            * **Zero-Centered Advantage:** Because outputs span negative and positive values, hidden layer activations maintain a mean close to zero, accelerating gradient descent convergence compared to Sigmoid.
            * **Stronger Gradients:** Peak derivative reaches $1.0$ at $x = 0$ ($4\\times$ larger than Sigmoid's $0.25$), making Tanh preferred over Sigmoid in recurrent architectures (RNNs/LSTMs).
            * **Limitation (Saturation):** Still suffers from vanishing gradients when $|x| > 4$, where $\\tanh(x) \\to \\pm 1$ and $\\tanh'(x) \\to 0$.
            """
        )


if __name__ == "__main__":
    main()
