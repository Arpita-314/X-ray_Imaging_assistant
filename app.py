# app.py
import streamlit as st
from main import generate_explanation_and_code
from simulation import generate_phantom, simulate_xray, plot_results

st.title("GPU X-ray Research Assistant")

query = st.text_input("Enter your query:")

if query:
    explanation, code_params = generate_explanation_and_code(query)
    st.subheader("Explanation")
    st.write(explanation)
    
    phantom = generate_phantom(**code_params)
    projection = simulate_xray(phantom)
    
    st.subheader("Simulation Results")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,2,figsize=(8,4))
    axs[0].imshow(phantom.cpu().numpy(), cmap='gray')
    axs[0].set_title("2D Phantom")
    axs[1].plot(projection.cpu().numpy())
    axs[1].set_title("X-ray Projection")
    st.pyplot(fig)
