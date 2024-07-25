import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('Pickle_files/laptop_price_model.pkl','rb'))
df = pickle.load(open('Pickle_files/laptop_price_df.pkl','rb'))

st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
Type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop(in Kgs)')

# Touchscreen
touchscreen = st.selectbox('TouchScreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.number_input('Screen Size(inches)')

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',df['CpuName'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['GpuBrand'].unique())

os = st.selectbox('OS',df['Os'].unique())

if st.button('Predict Price'):

    try:
        # query
        ppi = None
        if touchscreen == 'Yes':
            touchscreen = 1
        else:
            touchscreen = 0

        if ips == 'Yes':
            ips = 1
        else:
            ips = 0

        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])

        if screen_size <= 0:
            raise ValueError("Screen size must be greater than zero.")

        ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
        query = np.array([company,Type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

        query = query.reshape(1,12)
        st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))

    except ZeroDivisionError:
        st.error("Screen size must be greater than zero to calculate PPI.")
    except ValueError as ve:
        st.error(str(ve))
    except Exception as e:
        st.error(f"An error occurred: {e}")
