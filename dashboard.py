import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Judul dan deskripsi dashboard
st.title('Analisis Penggunaan Sepeda dalam Sistem Bike Sharing')
st.write('Dashboard ini melakukan analisis faktor-faktor yang mempengaruhi penggunaan sepeda dan dampak cuaca.')

# Unggah data
st.header('Unggah Data')
data = st.file_uploader('day.csv', type=['csv'])

# Jika data sudah diunggah, tampilkan data dan analisis
if data is not None:
    df = pd.read_csv(data)

    # Tampilkan data
    st.subheader('Data Bike Sharing')
    st.write(df)

    # Analisis faktor-faktor berpengaruh
    st.header('Analisis Faktor-Faktor')
    
    # Pilih variabel independen dan dependen
    X = df[['temp', 'registered', 'casual', 'instant']]
    y = df['cnt']

    # Bagi data menjadi data pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Latih model regresi linear
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Tampilkan faktor-faktor yang berpengaruh
    st.write('Koefisien Regresi:')
    st.write('temp:', model.coef_[0])
    st.write('registered:', model.coef_[1])
    st.write('casual:', model.coef_[2])
    st.write('instant:', model.coef_[3])
    # Konversi kolom 'tanggal' ke tipe data datetime
    df['dteday'] = pd.to_datetime(df['dteday'])

    # Set kolom 'tanggal' sebagai index
    df.set_index('dteday', inplace=True)

    # Resample data harian
    daily_data = df['cnt'].resample('D').sum()

    # Eksplorasi tren harian
    st.title("Eksplorasi Tren Harian Penggunaan Sepeda")

    # Plot tren harian
    st.write("Grafik Tren Harian:")
    st.line_chart(daily_data, use_container_width=True)

    # Eksplorasi musiman
    st.title("Eksplorasi Tren Musiman Penggunaan Sepeda")

    # Plot tren musiman (misalnya, bulanan)
    monthly_data = daily_data.resample('M').sum()
    st.write("Grafik Tren Musiman (mnth):")
    st.line_chart(monthly_data, use_container_width=True)

    # Eksplorasi dengan heatmap
    # Anda juga dapat mencoba heatmap untuk mengidentifikasi pola musiman berdasarkan hari dalam seminggu dan bulan dalam tahun
    df['weekday'] = df.index.day_name()
    df['mnth'] = df.index.month_name()

    heatmap_data = df.groupby(['weekday', 'mnth'])['cnt'].mean().unstack()
    st.write("Heatmap Tren Musiman:")
    st.write(heatmap_data.style.background_gradient(cmap='YlGnBu'))

    # Analisis dampak cuaca
    st.header('Analisis Dampak Cuaca')
    
    # Visualisasi jumlah peminjaman berdasarkan suhu
    plt.figure(figsize=(8, 6))
    plt.scatter(df['temp'], df['cnt'])
    plt.xlabel('temp')
    plt.ylabel('Jumlah Peminjaman')
    plt.title('Pengaruh Suhu terhadap Jumlah Peminjaman')
    st.pyplot(plt)

    # Visualisasi jumlah peminjaman berdasarkan hujan
    plt.figure(figsize=(8, 6))
    df['hum_bin'] = df['hum'].apply(lambda x: 'Hujan' if x == 1 else 'Tidak Hujan')
    df.groupby('hum_bin')['cnt'].mean().plot(kind='bar')
    plt.xlabel('hum')
    plt.ylabel('Rata-rata Jumlah Peminjaman')
    plt.title('Pengaruh Hujan terhadap Rata-rata Jumlah Peminjaman')
    st.pyplot(plt)

    # Visualisasi jumlah peminjaman berdasarkan kecepatan angin
    plt.figure(figsize=(8, 6))
    plt.scatter(df['windspeed'], df['cnt'])
    plt.xlabel('windspeed')
    plt.ylabel('Jumlah Peminjaman')
    plt.title('Pengaruh Kecepatan Angin terhadap Jumlah Peminjaman')
    st.pyplot(plt)

    # Analisis perbedaan antara pengguna terdaftar dan kasual
    st.header('Perbedaan Antara Pengguna Terdaftar dan Kasual')
    
    # Visualisasi perbedaan pengguna terdaftar dan kasual
    df.groupby('casual')['registered'].mean().plot(kind='bar')
    plt.xlabel('Tipe Pengguna')
    plt.ylabel('Rata-rata Jumlah Peminjaman')
    plt.title('Perbedaan Pola Peminjaman antara Pengguna Terdaftar dan Kasual')
    st.pyplot(plt)
