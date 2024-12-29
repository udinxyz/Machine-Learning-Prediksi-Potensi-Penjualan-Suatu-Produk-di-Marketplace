import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# Load the trained machine learning model
MODEL_PATH = 'optimized_model.pkl'  # Replace with the actual path
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# Streamlit App UI
def main():
    st.title("Prediksi Potensi Penjualan Produk")
    st.markdown("**Cek potensi produk Anda untuk terjual dengan sukses di marketplace!**")
    
    # Sidebar input fields
    st.sidebar.header("Masukkan Detail Produk")
    quantity = st.sidebar.slider("Jumlah (unit):", 1, 100, value=1)
    total_price = st.sidebar.number_input("Harga Total (dalam USD):", min_value=1.0, value=10.0, step=0.1)
    product_category = st.sidebar.selectbox(
        "Kategori Produk:",
        [
            "Olahraga & Luar Ruangan",
            "Rumah & Dapur",
            "Kecantikan & Kesehatan",
            "Buku",
            "Elektronik",
            "Pakaian",
        ]
    )
    payment_type = st.sidebar.selectbox(
        "Metode Pembayaran:",
        ["Kartu Kredit", "PayPal", "Kartu Debit", "Gift Card"]
    )
    payday_season = st.sidebar.checkbox("Tanggal Gajian Bang", value=False)
    is_high_price = st.sidebar.checkbox("Harga Tinggi Dari Pada Rata-Rata", value=False)

    # Preprocess user inputs
    def preprocess_input(quantity, total_price, product_category, payment_type, payday_season, is_high_price):
        avg_price_per_unit = total_price / quantity
        input_data = {
            "Quantity": quantity,
            "Total Price": total_price,
            "Avg Price Per Unit": avg_price_per_unit,
            "Payday Season": 1 if payday_season else 0,
            "Price Per Quantity": total_price / quantity,
            "Is High Price": 1 if is_high_price else 0,
            "Product Category_Books": 1 if product_category == "Buku" else 0,
            "Product Category_Clothing": 1 if product_category == "Pakaian" else 0,
            "Product Category_Electronics": 1 if product_category == "Elektronik" else 0,
            "Product Category_Home & Kitchen": 1 if product_category == "Rumah & Dapur" else 0,
            "Product Category_Sports & Outdoors": 1 if product_category == "Olahraga & Luar Ruangan" else 0,
            "Payment Type_Debit Card": 1 if payment_type == "Kartu Debit" else 0,
            "Payment Type_Gift Card": 1 if payment_type == "Gift Card" else 0,
            "Payment Type_PayPal": 1 if payment_type == "PayPal" else 0,
        }
        return pd.DataFrame([input_data])

    input_df = preprocess_input(quantity, total_price, product_category, payment_type, payday_season, is_high_price)

    # Predict and display results
    if st.button("Prediksi Potensi Penjualan"):
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        probability = probabilities[1] * 100

        # Risk Alert: Highlight risky products
        if probability < 30:
            st.error("⚠️ **Peringatan Risiko Tinggi** ⚠️")
            st.warning("Produk ini memiliki risiko sangat tinggi untuk tidak laku. Anda harus segera mempertimbangkan untuk menurunkan harga, memberikan diskon, atau menambahkan nilai jual unik (USP) pada produk.")

        # Display textual results
        if probability > 70:
            st.success(f"Produk ini memiliki potensi penjualan yang sangat baik dengan tingkat keyakinan {probability:.2f}%.")
            st.info("Strategi: Lanjutkan strategi Anda saat ini dan pertimbangkan untuk memperluas stok produk.")
        elif 50 <= probability <= 70:
            st.info(f"Produk ini memiliki potensi penjualan yang baik dengan tingkat keyakinan {probability:.2f}%.")
            st.warning("Strategi: Tingkatkan promosi produk dengan diskon atau deskripsi yang lebih menarik.")
        else:
            st.warning(f"Produk ini memiliki potensi penjualan yang rendah dengan tingkat keyakinan {probability:.2f}%.")
            st.error("Strategi: Kurangi harga atau tambahkan opsi bundling untuk meningkatkan daya tarik.")

        # Visualization: Interactive Probability Bar Chart
        st.subheader("Visualisasi Probabilitas Prediksi")
        categories = ['Tidak Laku', 'Laku']
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=[probabilities[0] * 100, probabilities[1] * 100],
                text=[f"{probabilities[0] * 100:.2f}%", f"{probabilities[1] * 100:.2f}%"],
                textposition='auto',
                marker_color=['#FF6F61', '#6B8E23']  # Custom colors
            )
        ])
        fig.update_layout(
            title="Probabilitas Penjualan",
            xaxis_title="Kategori",
            yaxis_title="Tingkat Keyakinan (%)",
            template="plotly_white",
            title_font_size=20
        )
        st.plotly_chart(fig)
        
    

if __name__ == "__main__":
    main()
