
import os
from flask import Flask, flash, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('HELLO WORLD')



UPLOAD_FOLDER = './Temp'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def fileUpload():
    target=os.path.join(UPLOAD_FOLDER,'test_docs')
    if not os.path.isdir(target):
        os.mkdir(target)
    logger.info("welcome to upload`")
    file = request.files['file'] 
    filename = secure_filename(file.filename)
    destination="/".join([target, filename])
    file.save(destination)
    session['uploadFilePath']=destination
    response= file
    return response

if __name__ == "__main__":
    app.secret_key = os.urandom(24)
    app.run(debug=True,use_reloader=False)


flask_cors.CORS(app, expose_headers='Authorization')


# from flask import Flask, render_template, request
# app = Flask(__name__)

# wsgi_app = app.wsgi_app

# @app.route('/', methods=['GET','POST'])
# def hello():
#     if request.method=="POST":
#         file = request.files['file']
#         file.save(os.path.join("uploads",file.filename))
#         return render_template("index.html",message="upload")
#     return render_template("index.html", messafe="upload")    

# if __name__ == "__main__":
#     import os
#     HOST = os.environ.get('SERVER_HOST', 'localhost') 
#     try:
#         PORT= int(os.environ.get('SERVER_POST','5000'))
#     except ValueError:
#         PORT = 5000
#     app.run(HOST,PORT)