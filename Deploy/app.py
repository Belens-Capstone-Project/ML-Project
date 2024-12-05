import pandas as pd
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Inisialisasi Flask
app = Flask(__name__)

# Konfigurasi upload file
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Maksimum ukuran file upload
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Load model
model = load_model('best_model (2).keras')

# Mapping label kategori
label_map = {
        0: 'ABC Kopi Susu',
        1: 'BearBrand',
        2: 'Benecol Lychee 100ml',
        3: 'Cimory Bebas Laktosa 250ml',
        4:'Cimory Susu Coklat Cashew',
        5: 'Cimory Yogurt Strawberry',
        6: 'Cola-Cola 390ml',
        7: 'Fanta Strawberry 390ml',
        8: 'Floridina 350ml',
        9: 'Fruit Tea Freeze 350ml',
        10: 'Hydro Coco Original 250ml',
        11: 'Ichitan Thai Green Tea',
        12: 'Larutan Penyegar rasa Jambu',
        13: 'Mizone 500ml',
        14: 'NU Green Tea Yogurt',
        15: 'Nutri Boost Orange Flavour 250ml',
        16: 'Pepsi Blue 330ml',
        17: 'Pocari Sweat 500 ml',
        18: 'Sprite 390ml',
        19: 'Tebs Sparkling 330ml',
        20: 'Teh Pucuk Harum',
        21: 'Teh Kotak 200ml',
        22: 'Tehbotol Sosro 250ml',
        23: 'Ultra Milk Coklat Ultrajaya 200ml',
        24: 'Ultramilk Fullcream 250ml',
        25: 'Yakult',
        26: 'You C 1000 Orange'
}

# Load file CSV untuk nilai gizi
gizi_df = pd.read_csv('Minuman.csv')

# Contoh data prediksi hasil
some_data = {
    "message": "Prediksi berhasil",
    "prediction": "",
    "confidence": 0.0,
    "gizi": {}
}


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Pastikan folder uploads ada
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        # Proses gambar dan prediksi
        img = Image.open(file_path)

        # Pastikan gambar dalam format RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = img.resize((120, 120))  # Resize image to match model input
        img_array = np.array(img) / 255.0  # Normalize image
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi
        prediction = model.predict(img_array)
        pred_class = np.argmax(prediction, axis=1)[0]
        predicted_label = label_map.get(pred_class, "Unknown")

        # Mencari nilai gizi dari CSV berdasarkan nama produk
        gizi_data = gizi_df[gizi_df['Nama Produk'] == predicted_label].iloc[0]
        gizi_info = gizi_data.to_dict()  # Mengonversi baris CSV menjadi dictionary

        # Update some_data dengan hasil prediksi dan nilai gizi
        some_data["prediction"] = predicted_label
        some_data["confidence"] = float(np.max(prediction) * 100)
        some_data["gizi"] = gizi_info

        # Return the prediction and nutrition information as JSON
        return jsonify({"status": "success", "data": some_data})


@app.route('/api/hasil', methods=['GET'])
def hasil_api():
    return jsonify({"status": "success", "data": some_data})


if __name__ == '__main__':
    app.run(debug=True)
