import os
import warnings
warnings.filterwarnings("ignore")
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

#app.py
from flask import Flask, flash, request, redirect, url_for, render_template
#import urllib.request
#import os
from werkzeug.utils import secure_filename
 
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        
        def image_to_caption(img_path):
            vgg_model = tf.keras.models.load_model('vgg16.h5')
        
            # loading tokenizer
            with open('tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
            
            def idx_to_word(integer, tokenizer):
                for word, index in tokenizer.word_index.items():
                    if index == integer:
                        return word
                return None
            
            def predict_caption(model, image, tokenizer, max_length):
                # add start tag for generation process
                in_text = 'startseq'
                # iterate over the max length of sequence
                for i in range(max_length):
                    # encode input sequence
                    sequence = tokenizer.texts_to_sequences([in_text])[0]
                    # pad the sequence
                    sequence = pad_sequences([sequence], max_length)
                    # predict next word
                    yhat = model.predict([image, sequence], verbose=0)
                    # get index with high probability
                    yhat = np.argmax(yhat)
                    # convert index to word
                    word = idx_to_word(yhat, tokenizer)
                    # stop if word not found
                    if word is None:
                        break
                    # append word as input for generating next word
                    in_text += " " + word
                    # stop if we reach end tag
                    if word == 'endseq':
                        break
                return in_text
            
            
            #directory = 'Images'
            # load the image from file
            #img_path = directory + '/' + '1001773457_577c3a7d70.jpg'
            image = load_img(img_path, target_size=(224, 224))
            # convert image pixels to numpy array
            image = img_to_array(image)
            # reshape data for model
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            # preprocess image for vgg
            image = preprocess_input(image)
            # extract features
            feature = vgg_model.predict(image, verbose=0)
            
            model = tf.keras.models.load_model('best_model.h5')
            
            y_pred = predict_caption(model, feature, tokenizer, max_length=35)
            y_pred = y_pred.split(' ')
            y_pred = ' '.join(y_pred[1:len(y_pred)-1])
            
            return y_pred
        #print(image_to_caption('Images/1001773457_577c3a7d70.jpg'))
        img_dir = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        caption = image_to_caption(img_dir)
        flash("The given Image caption is: ")
        flash(caption)
        #print(img_dir,"*"*50)
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == "__main__":
    app.run()