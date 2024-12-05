import pandas as pd
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
import tensorflow as tf

# Inisialisasi Flask
app = Flask(__name__)

# Konfigurasi upload file
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Maksimum ukuran file upload
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Load model
tflite_interpreter = tf.lite.Interpreter(model_path="best_model_optimized.tflite")
tflite_interpreter.allocate_tensors()

# Mapping label kategori
label_map = {
        0: 'ABC Kopi Susu',
        1: 'BearBrand',
        2: 'Benecol Lychee 100ml',
        3: 'Cimory Bebas Laktosa 250ml',
        4: 'Cimory Susu Coklat Cashew',
        5: 'Cimory Yogurt Strawberry',
        6: 'Cola-Cola 390ml',
        7: 'Fanta Strawberry 390ml',
        8: 'Floridina 350ml',
        9: 'Fruit Tea Freeze 350ml',
        10: 'Garantea',
        11: 'Golda Cappucino',
        12: 'Hydro Coco Original 250ml',
        13: 'Ichitan Thai Green Tea',
        14: 'Larutan Penyegar Rasa Jambu',
        15: 'Mizone 500ml',
        16: 'NU Green Tea Yogurt',
        17: 'Nutri Boost Orange Flavour 250ml',
        18: 'Oatside Cokelat',
        19: 'Pepsi Blue Kaleng',
        20: 'Pocari Sweat 500ml',
        21: 'Sprite 390ml',
        22: 'Tebs Sparkling 330ml',
        23: 'Teh Pucuk Harum',
        24: 'Teh Kotak 200ml',
        25: 'Tehbotol Sosro 250ml',
        26: 'Ultra Milk Coklat Ultrajaya 200ml',
        27: 'Ultramilk Fullcream 250ml',
        28: 'Yakult',
        29: 'You C 1000 Orange'
}


# Load file CSV untuk nilai gizi
csv_path = 'nutrisi.csv'
gizi_df = pd.read_csv(csv_path)

# Kolom gizi yang ingin diambil
columns_of_interest = [
    'total_energi', 'gula', 'lemak_jenuh', 'garam', 'protein', 'grade', 'rekomendasi'
]


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Pastikan folder uploads ada
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Proses gambar dan prediksi
        img = Image.open(file_path)

        # Pastikan gambar dalam format RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = img.resize((120, 120))  # Resize image to match model input
        img_array = np.array(img) / 255.0  # Normalize image
        img_array = np.expand_dims(img_array, axis=0)

        # Inference with TFLite
        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()

        # Set the input tensor
        tflite_interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))

        # Run the inference
        tflite_interpreter.invoke()

        # Get the prediction result
        prediction = tflite_interpreter.get_tensor(output_details[0]['index'])
        pred_class = np.argmax(prediction, axis=1)[0]
        predicted_label = label_map.get(pred_class, "Unknown")

        # Mencari nilai gizi dari CSV berdasarkan nama produk
        gizi_data = gizi_df[gizi_df['nama_produk'] == predicted_label]

        if gizi_data.empty:
            return jsonify({
                "status": "error",
                "message": f"Data nilai gizi untuk {predicted_label} tidak ditemukan."
            }), 404

        gizi_info = gizi_data[columns_of_interest].iloc[0].to_dict()

        # Return JSON
        return jsonify({
            "status": "success",
            "data": {
                "prediction": predicted_label,
                "confidence": float(np.max(prediction) * 100),
                "gizi": gizi_info
            }
        })


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
