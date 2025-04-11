import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Beam Analysis: Shear, Moment, Deflection & Rotation")

# User Inputs
st.header("User Inputs")

P1 = st.number_input("Enter load P1 (lb):", value=100.0)
P2 = st.number_input("Enter load P2 (lb):", value=150.0)
P3 = st.number_input("Enter load P3 (lb):", value=200.0)

a = st.number_input("Enter distance a (in) for load P1:", value=10.0)
b = st.number_input("Enter distance b (in) for load P2:", value=20.0)
c = st.number_input("Enter distance c (in) for load P3:", value=30.0)

L = st.number_input("Enter span L (in):", value=40.0)

X = st.number_input("Enter point X (in) where deflection and rotation are to be calculated:", value=25.0)

E = st.number_input("Enter modulus of elasticity E (psi):", value=29000000.0)
I = st.number_input("Enter moment of inertia I (in^4):", value=100.0)

if st.button("Run Analysis"):

    # Reaction Forces Calculation
    Rb = (P1 * a + P2 * b + P3 * c) / L
    Ra = (P1 + P2 + P3) - Rb

    # Shear Force and Bending Moment Diagrams
    dx = L / 10000.0
    x = np.arange(0, L + dx, dx)
    M = np.zeros_like(x)
    V = np.zeros_like(x)

    for i, xi in enumerate(x):
        if xi <= a:
            M[i] = Ra * xi
            V[i] = Ra
        elif xi <= b:
            M[i] = Ra * xi - P1 * (xi - a)
            V[i] = Ra - P1
        elif xi <= c:
            M[i] = Ra * xi - P1 * (xi - a) - P2 * (xi - b)
            V[i] = Ra - P1 - P2
        else:
            M[i] = Ra * xi - P1 * (xi - a) - P2 * (xi - b) - P3 * (xi - c)
            V[i] = Ra - P1 - P2 - P3

    # Plotting the diagrams (unchanged)
    plt.figure(figsize=(10, 8))

    # Bending Moment Diagram
    plt.subplot(2, 1, 1)
    plt.plot(x, M, linewidth=3)
    plt.grid(True)
    plt.title("Bending Moment Diagram")
    plt.xlabel("Distance from Left Support [in]")
    plt.ylabel("Bending Moment [lb-in]")

    # Shear Force Diagram
    plt.subplot(2, 1, 2)
    plt.plot(x, V, 'r', linewidth=3)
    plt.grid(True)
    plt.title("Shear Force Diagram")
    plt.xlabel("Distance from Left Support [in]")
    plt.ylabel("Shear Force [lb]")

    plt.tight_layout()
    st.pyplot(plt)  # Display the figure in Streamlit

    # Deflection and Rotation Using Virtual Work
    m_t = np.zeros_like(x)
    m_t_deriv = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi <= X:
            m_t[i] = xi * (L - X) / L
            m_t_deriv[i] = (L - X) / L
        else:
            m_t[i] = X * (L - xi) / L
            m_t_deriv[i] = -X / L

    delta = np.trapz(M * m_t, x) / (E * I)
    theta = np.trapz(M * m_t_deriv, x) / (E * I)

    # Display Results
    st.subheader("--- Results at x = {:.2f} in ---".format(X))
    st.write("**Deflection, Δ = {:.6e} in**".format(delta))
    st.write("**Rotation, θ  = {:.6e} rad**".format(theta))
