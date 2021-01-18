from keras.models import load_model
import numpy as np

def validateModel(X_val):
	'''
	   Validate the performance of the Model by loading the trained model weights.
	   Args:
	       X_val : Array for features.

	   Returns: Model prediction against the input features.
	'''
	Model_Loaded = load_model('best_model_22.h5')

	yhat_probs = Model_Loaded.predict(X_val, verbose=1)
	yhat_probs = yhat_probs.reshape(yhat_probs.shape[0],yhat_probs.shape[2])
	yhat_classes =np.argmax(yhat_probs,axis=1)

	return yhat_classes