from flask import Flask, render_template, request
import keras
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import load_img
from keras.applications.efficientnet import preprocess_input
from keras.preprocessing import image
import os
import efficientnet.tfkeras as efn

app = Flask(__name__, template_folder='templates', static_folder='css')
model = load_model('modelDR.h5')
target_img = os.path.join(os.getcwd(), 'css/images')


@app.route('/')
def index():
    return render_template('index.html')


ALLOWED_EXT = set(['jpg', 'png', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXT


def read_img(filename):
    img = load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


@app.route('/protect', methods=["GET", "POST"])
def protect():
    file = request.files['img']
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join('css/images', filename)
        file.save(file_path)
        img = read_img(file_path)
        pred = model.predict(img)
        pred = np.argmax(pred, axis=1)
        if pred[0] == 2:
            data = "NO DIABETIC RETINOPATHY"
        elif pred[0] == 0:
            data = "MILD DIABETIC RETINOPATHY"
        elif pred[0] == 1:
            data = "MODERATE DIABETIC RETINOPATHY"
        elif pred[0] == 4:
            data = "SEVERE DIABETIC RETINOPATHY"
        elif pred[0] == 3:
            data = "PROLIFERATIVE DIABETIC RETINOPATHY"
        return render_template('protect.html', data=data)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
