# from keras.applications import ResNet50
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import os
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, PasswordField, SubmitField, BooleanField, TextAreaField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError

from flask_sqlalchemy import SQLAlchemy
from flask import render_template, redirect
from flask import request, url_for, render_template, redirect
import io
import tensorflow as tf
import os
#ignore AVX AVX2 warning 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
UPLOAD_FOLDER = os.path.join(app.root_path ,'static','img')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)


class RegistrationForm(FlaskForm):
    username = StringField('Username',
                           validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
   
    gender = StringField('Gender',
                           validators=[DataRequired(), Length(min=2, max=20)])
    
    Report_result=StringField('Report_result',
                           validators=[DataRequired(), Length(min=1, max=20)])    
    Daily_activity=StringField('Daily_activity',
                           validators=[DataRequired(), Length(min=2, max=200)])                                                                     
    submit = SubmitField('Sign Up')
   
    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('That username is taken. Please choose a different one.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('That email is taken. Please choose a different one.')
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    
    gender =db.Column(db.String(60), nullable=False)
    Report_result=db.Column(db.String(60), nullable=False)
    Daily_activity=db.Column(db.String(60), nullable=False)

def load_model_():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
    global model
    model = load_model("VGG16-100-0.0001-adam.h5")
    

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	# image = imagenet_utils.preprocess_input(image)

	# return the processed image
	return image

    
@app.route("/", methods=["POST","GET"])
def predict():
    form = RegistrationForm()
	# initialize the data dictionary that will be returned from the
	# view
    data = {"success": "Upload X-ray"}
    title = "Upload an image"
    name = "default.png"
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            graph = tf.get_default_graph()
            model = load_model("VGG16-100-0.0001-adam.h5")
            image1 = flask.request.files["image"]
            # save the image to the upload folder, for display on the webpage.
            image = image1.save(os.path.join(app.config['UPLOAD_FOLDER'], image1.filename))
            
            # read the image in PIL format
            with open(os.path.join(app.config['UPLOAD_FOLDER'], image1.filename), 'rb') as f:
                image = Image.open(io.BytesIO(f.read()))
            
            # preprocess the image and prepare it for classification
            processed_image = prepare_image(image, target=(224, 224))

            with graph.as_default():
                preds = model.predict(processed_image)
            pred = np.argmax(preds,axis=1)[0]


            #Metrics
            accuracy = 0.7916 * 100


            data["accuarcy"] = accuracy
            if pred:
                data["prediction"] = "Positive"
            else:
                data["prediction"] = "Negative"

            data["success"] = "Uploaded"
            title = "predict"
            user = User(username=form.username.data,
                        email=form.email.data,
                        gender=form.gender.data,
                        Report_result= data["prediction"],
                        Daily_activity=form.Daily_activity.data
                        )
            db.session.add(user)
            db.session.commit()
            return render_template('index.html', data=data, title = title, name=image1.filename,form=form)
	# return the data dictionary as a JSON response
    return render_template('index.html', data = data, title=title, name=name,form=form)
# if this is the main thread of execution first load the model and
# then start the server
@app.route('/register',methods=["POST","GET"])
def reg():
    user=User.query.all()
   
    return render_template('report.html', title='Register', user=user)
@app.route('/presenation')
def presentation():
	return render_template('presentation.html')

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started.(60sec)"))
    # load_model_()
    # global graph
    # graph = tf.get_default_graph()
    app.run(debug=True)
  