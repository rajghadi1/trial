from flask import Flask, request, render_template, flash, redirect, url_for
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#from keras.models import load_model
from tensorflow.keras.models import load_model
import os
import io
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from werkzeug.utils import secure_filename


app = Flask(__name__)
model = pickle.load(open('beproject.pkl', 'rb'))
model2=pickle.load(open("heart.pkl",'rb'))
model3 = pickle.load(open('liver.pkl', 'rb'))
model4=pickle.load(open("cancer.pkl",'rb'))
model5=pickle.load(open("diabetes.pkl",'rb'))
model7=pickle.load(open("kidneyPKL.pkl",'rb'))

#with tf.device('/cpu:0'):
model6=load_model("tumor")
model8=load_model("tb")
model9 = load_model('covid')

@app.route('/')

@app.route('/home')
def home():
    return render_template('start.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/page')
def page():
    return render_template('page.html')

@app.route('/predict', methods=['POST'])
def predict():
 if request.method == 'POST':
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    output = model.predict(features_value)
    if output == 1:
        return render_template('detected.html', predict_text='Parkinson’s Disease Detected')
    elif output == 0:
        return render_template('notdetected.html', predict_text='Parkinson’s Disease Not Detected')
    else:
        return render_template('page.html', predict_text='Something went wrong')

@app.route('/heart')
def heart():
    return render_template('heart.html')

@app.route('/resultH', methods=['POST'])
def resultH():
    if request.method == 'POST':
        input_features2 = [float(x) for x in request.form.values()]
        features_value2 = np.array(input_features2)
        output2 = model2.predict([features_value2])

        if output2 == 1:
            return render_template('heart.html', predict_text='Heart Disease Detected')
        elif output2 == 0:
            return render_template('heart.html', predict_text='Heart Disease Not Detected')
        else:
            return render_template('heart.html', predict_text='Something went wrong')




@app.route('/liver')
def liver():
    return render_template('liver.html')

@app.route('/resultL', methods=['POST'])
def resultL():
    if request.method == 'POST':
        input_features3 = [float(x) for x in request.form.values()]
        features_value3 = np.array(input_features3)
        output3 = model3.predict([features_value3])
        if output3 == 1:
            return render_template('liver.html', predict_text='liver Disease Detected')
        elif output3 == 2:
            return render_template('liver.html', predict_text='liver Disease Not Detected')
        else:
            return render_template('liver.html', predict_text='Something went wrong')

@app.route('/sugar')
def sugar():
    return render_template('sugar.html')

@app.route('/resultD', methods=['POST'])
def resultD():
    if request.method == 'POST':
        input_features5 = [float(x) for x in request.form.values()]
        features_value5 = np.array(input_features5)
        output5 = model5.predict([features_value5])
        if output5 == 0:
            return render_template('sugar.html', predict_text='Diabetes Detected')
        elif output5 == 1:
            return render_template('sugar.html', predict_text='Diabetes Not Detected')
        else:
            return render_template('suagr.html', predict_text='Something went wrong')

@app.route('/cancer')
def cancer():
    return render_template('cancer.html')

@app.route('/resultC', methods=['POST'])
def resultC():
    if request.method == 'POST':
        input_features4 = [float(x) for x in request.form.values()]
        features_value4 = np.array(input_features4)
        output4 = model4.predict([features_value4])
        print(output4)
        if output4 == 0:
            return render_template('cancer.html', predict_text='Benign Tumor Detected')
        elif output4 == 1:
            return render_template('cancer.html', predict_text='Malignant Tumor Detected')
        else:
            return render_template('cancer.html', predict_text='Something went wrong')





def pred_covid(xray):
    test_image = load_img(xray, target_size=(200, 200), color_mode="grayscale")  # load image
    print("@@ Got Image for prediction")

    test_image = img_to_array(test_image) / 255  # convert image to np array and normalize
    test_image = np.expand_dims(test_image, axis=0)  # change dimention 3D to 4D

    result = model9.predict(test_image).round(3)  # predict diseased palnt or not
    print('@@ Raw result = ', result)

    pred = np.round(result)  # get the index of max value
    print(pred)
    if pred == 0:
        return "NORMAL"
    elif pred == 1:
        return 'covid detected'

    else:
        return "Something went wrong"

app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

@app.route('/covid')
def covid():
    return render_template('covid.html')

@app.route('/file_not_found')
def fnf():
    return render_template('covid.html', message="Please Select a Image First!!")

@app.route('/resultP', methods=['POST','GET'])
def resultP():
    if request.method == 'POST':
        file = request.files['image']  # fet input
        filename = file.filename
        print("@@ Input posted = ", filename)


        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for("fnf"))


        file_path = os.path.join('static/user_uploaded/', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred= pred_covid(xray=file_path)

        return render_template('covidresult.html', pred_output=pred, user_image=file_path)

        # check if the post request has the file part

@app.route('/kidney')
def kidney():
    return render_template('kidney.html')

@app.route('/resultK', methods=['POST'])
def resultK():
    if request.method == 'POST':
        input_features5 = [float(x) for x in request.form.values()]
        features_value5 = np.array(input_features5)
        output5 = model7.predict([features_value5])
        print(output5)
        if output5 == 0:
            return render_template('kidney.html', predict_text='Chronic KIdney Disease Not Detected')
        elif output5 == 1:
            return render_template('kidney.html', predict_text='Chronic KIdney Disease Detected')
        else:
            return render_template('kidney.html', predict_text='Something went wrong')



def pred_tumor(brain):
    test_image = load_img(brain, target_size=(200, 200), color_mode="grayscale")  # load image
    print("@@ Got Image for prediction")

    test_image = img_to_array(test_image) / 255  # convert image to np array and normalize
    test_image = np.expand_dims(test_image, axis=0)  # change dimention 3D to 4D

    result = model6.predict(test_image).round(3)  # predict diseased palnt or not
    print('@@ Raw result = ', result)

    pred = np.round(result)  # get the index of max value
    print(pred)
    if pred == 0:
        return "NORMAL"
    elif pred == 1:
        return 'Brain Tumor detected'

    else:
        return "Something went wrong"

app.config['SECRET_KEY'] = '57628bb0b13ce0c676dfde280ba245'

@app.route('/brain')
def brain():
    return render_template('brain.html')

@app.route('/file_not_found2')
def fnf2():
    return render_template('brain.html', message="Please Select a Image First!!")

@app.route('/resultB', methods=['POST','GET'])
def resultB():
    if request.method == 'POST':
        file = request.files['image']  # fet input
        filename = file.filename
        print("@@ Input posted = ", filename)


        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for("fnf2"))


        file_path = os.path.join('static/user_uploaded/', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred= pred_tumor(brain=file_path)

        return render_template('tumorresult.html', pred_output=pred, user_image=file_path)

        # check if the post request has the file part

def pred_tb(tb):
    test_image3 = load_img(tb, target_size=(200, 200), color_mode="grayscale")  # load image
    print("@@ Got Image for prediction")

    test_image3 = img_to_array(test_image3) / 255  # convert image to np array and normalize
    test_image3 = np.expand_dims(test_image3, axis=0)  # change dimention 3D to 4D

    result3 = model8.predict(test_image3).round(3)  # predict diseased palnt or not
    print('@@ Raw result = ', result3)

    pred3 = np.round(result3)  # get the index of max value
    print(pred3)
    if pred3 == 0:
        return "NORMAL"
    elif pred3 == 1:
        return 'TB detected'

    else:
        return "Something went wrong"

app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280245'

@app.route('/tb')
def tb():
    return render_template('tb.html')

@app.route('/file_not_found3')
def fnf3():
    return render_template('tb.html', message="Please Select a Image First!!")

@app.route('/resultT', methods=['POST','GET'])
def resultT():
    if request.method == 'POST':
        file = request.files['image']  # fet input
        filename = file.filename
        print("@@ Input posted = ", filename)


        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for("fnf3"))


        file_path = os.path.join('static/user_uploaded/', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred3= pred_tb(tb=file_path)

        return render_template('tbresult.html', pred_output=pred3, user_image=file_path)

        # check if the post request has the file part



if __name__ == '__main__':
    app.run(debug=True)
