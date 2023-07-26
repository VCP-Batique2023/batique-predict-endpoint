from firebase_functions import https_fn
from firebase_admin import initialize_app

import numpy as np
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

initialize_app()

model = load_model('model/keras_model.h5')

labels =  {
    0 : True,
    1 : False,
}

@https_fn.on_request()
def predict(req: https_fn.Request) -> https_fn.Response:
    try:
        print(req)
        image_data = req.files['file']
        image_bytes = image_data.read()

        img = image.load_img(io.BytesIO(image_bytes), target_size=(224, 224))
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        predicted_probs = model.predict(x)[0]
        predicted_class_index = np.argmax(predicted_probs)
        
        response = labels[predicted_class_index]
        return https_fn.Response(response), 200
    
    except Exception as e:
        return https_fn.Response({'error': str(e)}), 500