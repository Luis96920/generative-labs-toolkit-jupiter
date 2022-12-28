from crypt import methods
from flask_bootstrap import Bootstrap5, Bootstrap4
from flask import Flask, jsonify, render_template, request, url_for, flash, redirect, session, make_response
import os
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import pyrebase
from utils import *
import datetime
import json

import re
from io import BytesIO
from PIL import Image





## APP Config
app = Flask(__name__)
static_images = os.path.join('static', 'images')
app.config['STATIC_IMAGES'] = static_images
app.config['TEMPLATES_AUTO_RELOAD'] = True

## Change for production to utils.secret_manager
app.config['SECRET_KEY'] = "CZRzu8dbqxlYKasil0QJqezaS7Z2cXJ1ydzrfrXs0BVT5mfMpKQFAm54pX2dbZudrwX4k7AcK61fhjtIwBJ_5A"
bootstrap = Bootstrap5(app)

##  DB -- Need Creds
cred = credentials.Certificate("firestore_token.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


## Firebase AUTH creds -- Change for production to utils.secret_manager
with open ("firebaseConfig.json") as json_file:
    firebaseConfig = json.load(json_file)

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()


## tx2 Model Params
size = [512, 576, 640, 704, 768, 832, 896, 960, 1042]
scale = [x  for x in range(5,105, 5)]
ddim_step = [x for x in range(0,100,10)]
samplers = ['ddim', "plms", "k_euler", "k_euler_ancestral", "k_heun", "k_dpm_2", "k_dpm_2_ancestral", "k_lms" ]



##### LOGIN - AUTH #####

@app.route("/login" ,methods=('GET', 'POST'))
def login(alerts=None):
    if ("user" in session):
        return redirect(url_for("home"))

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        try:
            auth.sign_in_with_email_and_password(email, password)
            session['user'] =  email
            #### funcion ON -- > INSTANCIA -- CHECKEAR
            # IF VM is not running, start it, loading animation  (loading true --> JS loading )
            if connect_vm('get_status').get('status') != "RUNNING":
                connect_vm("start")
                return redirect(url_for("home"), loading=True)
            return redirect(url_for("home"))

        except:
            alerts = "Bad credentials"
            return redirect(url_for("login", alerts=alerts))

    return render_template("login.html", alerts=alerts)

@app.route("/logout" ,methods=('GET', 'POST'))
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/register" ,methods=('GET', 'POST'))
def register():
    return render_template("register.html")

@app.route("/passrecover" ,methods=('GET', 'POST'))
def passrecover():
    return render_template("passrecover.html")

@app.route("/sendrecover" ,methods=('GET', 'POST'))
def sendrecover():
    email = request.get_json().get("email")
    print(email)
    try:
        auth.send_password_reset_email(email)
        print("OK")
        return jsonify("", render_template("success_email_recover.html", email=email))
    except:
        print("NO EMAIL")
        return jsonify("", render_template("error_email_recover.html", email=email))
    

@app.route('/authenticated', methods=['POST','GET'])
def authenticated():
    # Global variable to check anytime whether the user is authenticated or not
    if request.method == "POST":
        # IF VM is not running, start it, loading animation  (loading true --> JS loading )
        if connect_vm('get_status').get('status') != "RUNNING":
            connect_vm("start")
            return redirect(url_for("home"), loading=True)
        return redirect(url_for('home'))

    if request.method == "GET":
        for var in ['user','email','photo']:
            session[var] = request.cookies.get(var)
        if ("user" not in session):
            return redirect(url_for('login'), alerts="Usuario no encontrado")
        
        # IF VM is not running, start it, loading animation  (loading true --> JS loading )
        if connect_vm('get_status').get('status') != "RUNNING":
            connect_vm("start")
            return render_template("home.html", loading=True)
        return redirect(url_for('home'))


#### ROUTES ####
@app.route("/inspire" ,methods=('GET', 'POST'))
def inspire():
    if ("user" not in session):
        return redirect(url_for("login"))

    # TODO ### Armar Dataclass de MODELO con params
    context = {
        "sampler" : samplers,
        "size" : {
            "values" : size,
            "min": min(size),
            "max": max(size),
            "step" : size[1]-size[0]
            },
        "n_samples":[1,2,4],
        "n_iters" : [1,2,4],
        "scale" : {
            "values" : scale,
            "min": min(scale),
            "max": max(scale),
            "step" : scale[1]-scale[0]
                },
        "ddim_step" : {
            'values' : ddim_step,
            "min": min(ddim_step),
            "max": max(ddim_step),
            "step" : ddim_step[1]-ddim_step[0]
                }     
    }

    full_filename = os.path.join(app.config['STATIC_IMAGES'], 'creative_labs.png')
    return render_template("txt2image.html", img=full_filename, context = context)

@app.route("/get_images", methods = ['POST'])
def get_img():
    job_id = request.get_json().get('job_id')
    job_type = request.get_json().get("job_type")
    print(job_type)
    job_data = db.collection("job_collection").document(job_id).get().to_dict()
    
    #hardcoded until have a responsive display for grid
    grid = None
    
    if job_data.get('url_to_image_grid') and grid:
        image_to_show = job_data.get('url_to_image_grid')
        full_filename = f"https://storage.googleapis.com/genlabs_images_bucket/outputs/{job_type}-samples/{image_to_show}" 

    else:
        image_to_show = job_data.get('urls_to_images')[0]
        full_filename = f"https://storage.googleapis.com/genlabs_images_bucket/outputs/{job_type}-samples/samples/{image_to_show}"    
    return jsonify("", render_template("get_images.html", img=full_filename))


@app.route("/submit_job", methods = ['GET', 'POST'])
def submit_job():
    request_time = datetime.datetime.now()
    params = request.get_json()

    user_id = session['user'] # Change for User email - Cloud Identity
    job_type = params.get("job_type", None)
    data = {
        "user" : user_id,
        "params" : params,
        'created_at' : request_time,
        'job_type' : job_type
    }
    print(params.get('strenght'))
    if params.get("imageBase64"):
        image_data = re.sub('^data:image/.+;base64,', '', params.get('imageBase64'))
        data['init-img'] = image_data

    job_id = db.collection("job_collection").add(data)[1].id
    send_job(job_type, data, job_id, db)
    print(job_id)
    return make_response(jsonify({"job_id" : job_id, "job_type":job_type}))


@app.route('/my_gallery', methods=['GET', 'POST'])
def my_gallery():
    if ("user" not in session):
        return redirect(url_for("login"))

    """
    Aca hay que ver como hacer con los prompts y las imagenes con varias samples - Hacer diccionario prompt:[imagenes]
    """
    results = db.collection("job_collection").where("user", "==", session['user']).get()
    jobs = [ x.to_dict() for x in results]

    if len(jobs) == 0:
        return render_template("my_gallery.html", no_image="no_image")

    # TODO REDIS??? -- Resize????   
    urls = np.array([jobs[x].get('urls_to_images', ['empty'])[0] for x  in range(len(jobs))]).flatten()
    prompts = np.array([jobs[x].get('params', {}).get('prompt', "") for x  in range(len(jobs))]).flatten()
    job_type = np.array([jobs[x].get('job_type', "txt2img") for x  in range(len(jobs))]).flatten()

    print(job_type)

    prompt_0 = prompts[0]
    prompts = prompts[1:]
    url_0 = urls[0]
    urls = urls[1:]
    job_type_0 = job_type[0]
    job_types = job_type[1:]
    first_img = [prompt_0, url_0, job_type_0]
    all_img = zip(prompts, urls, job_types)

    return render_template("my_gallery.html", first_img=first_img, all_img=all_img)


# Mejorar HTML - Static Images
@app.route("/prompt_guide" ,methods=('GET', 'POST'))
def prompt_guide():
    if ("user" not in session):
        return redirect(url_for("login"))
    """
    Tomar lo que dice y reformularlo por que es un choreo.
    """
    return render_template("prompt_guide.html")

@app.route("/" ,methods=('GET', 'POST'))
def home():
    if ("user" not in session):
        return redirect(url_for("login"))
    return render_template("home.html")


@app.route("/img2img", methods=("GET", "POST"))
def img2img():
    if ("user" not in session):
        return redirect(url_for("login"))

    img_to_edit = "static/images/blank_image.png"
    data = {
        "img_to_edit":img_to_edit,
    }
    return render_template("img2img.html", data=json.dumps(data))

### DEV -------

@app.route("/inpainting", methods=("GET", "POST"))
def inpainting():
    if ("user" not in session):
        return redirect(url_for("login"))

    img_to_edit = "static/images/blank_image.png"
    data = {
        "img_to_edit":img_to_edit,
    }
    return render_template("inpainting.html", data=json.dumps(data))

if __name__=="__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
