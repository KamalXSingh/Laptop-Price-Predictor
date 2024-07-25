import streamlit as st
import pickle
import numpy as np

# Loading the model and data
pipe = pickle.load(open('Pickle_files/laptop_price_model.pkl', 'rb'))
df = pickle.load(open('Pickle_files/laptop_price_df.pkl', 'rb'))

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("Laptop Price Predictor")
st.markdown("""
    Welcome to the Laptop Price Predictor! Fill out the form below with the specifications of your desired laptop.
    Our model will predict the estimated price based on your inputs.
    """)

# Layout
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('Brand', ["None"]+ list(df['Company'].unique()))
    Type = st.selectbox('Type', ["None"] + list(df['TypeName'].unique()),index=0)
    ram = st.selectbox('RAM (in GB)', ["None"] + [2, 4, 6, 8, 12, 16, 24, 32, 64])
    weight = st.number_input('Weight of the Laptop (in Kgs)', min_value=0.0, step=0.1)
    touchscreen = st.selectbox('Touchscreen', [ 'No', 'Yes'])
    ips = st.selectbox('IPS', [ 'No', 'Yes'])

with col2:
    screen_size = st.number_input('Screen Size (in inches)', min_value=0.0, step=0.1)
    resolution = st.selectbox('Screen Resolution',
                              ['None', '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800',
                               '2560x1600', '2560x1440', '2304x1440'])
    cpu = st.selectbox('CPU', ["None"] + list(df['CpuName'].unique()))
    hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
    ssd = st.selectbox('SSD (in GB)',  [0, 8, 128, 256, 512, 1024])
    gpu = st.selectbox('GPU', ["None"] + list(df['GpuBrand'].unique()))
    os = st.selectbox('OS', ["None"] + list(df['Os'].unique()))

if st.button('Predict Price'):
    try:
        # Ensuring all fields are selected
        if "None" in [company, Type, ram, touchscreen, ips, resolution, cpu, hdd, ssd, gpu, os] or screen_size == 0:
            st.error("Please fill out all fields before predicting.")
        else:
            # Prepare the query
            touchscreen = 1 if touchscreen == 'Yes' else 0
            ips = 1 if ips == 'Yes' else 0

            X_res, Y_res = map(int, resolution.split('x'))

            ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
            query = np.array([company, Type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
            query = query.reshape(1, 12)

            # Price Prediction
            predicted_price = int(np.exp(pipe.predict(query)[0]))

            st.success(f"The predicted price of this configuration is $ {predicted_price:,}")
    except ZeroDivisionError:
        st.error("Screen size must be greater than zero to calculate PPI.")
    except ValueError as ve:
        st.error(str(ve))
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
