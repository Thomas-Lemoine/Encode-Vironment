import os
from utils.settings import IMAGES_PATH
from utils.functions import get_latent_repr, get_recon_img
from flask import Flask, render_template, request, redirect, url_for, abort
from werkzeug.utils import secure_filename
import utils.models as models

DATA_PATH = 'data'
IMAGES_PATH = 'static'

# Load model
model = models.load_model()

app = Flask(__name__)
#app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024					#max allowed file size
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg']		#acceptable extensions						#save to this folder
			
@app.route("/")
def index():

    return render_template('index.html')

# i need to set an accept attribute in html
# create another upload system for decoding


@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if uploaded_file.filename != '':						#if filename is not empty
        file_ext = os.path.splitext(filename)[1]			#judging the file extension
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            abort(400)
        filename = f"uploaded_image.{file_ext}"
        uploaded_file.save(os.path.join(IMAGES_PATH, filename)) #if file extension is acceptable then save it to uploads directory
        

        # bring the image through the model and get the reconstructed image, 
        # and the latent representation of the image (tensor & np)
        #latent, latent_np = get_latent_repr(filename, model)
        #recon_img = get_recon_img(latent, model)
        #recon_img_name = 'recon_img.png'
        recon_img = upload_file
        recon_img.save(os.path.join(IMAGES_PATH, recon_img_name))
        

        # show the slider, with original values being the latent representation
        # the slider goes from -5 to 5 by default. If the abs() of any of the values of the latent
        # representation are above 5, make the slider range be between those values. 
        # make the step be 100 values

        # add a button with "save compressed_data" to save the latent values into a format
        # that can be decoded later on

    #return redirect(url_for('index'))
    return render_template('uploaded_img.html', img_name = filename, recon_img_name = recon_img_name)

#new features :
# A playground for changing the latent variables with a slider


if __name__ == '__main__':
	app.run(debug=True)