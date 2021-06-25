from keras import backend as K
from model import InstantiateModel
from keras.models import Model
from keras.optimizers import Adamax
from keras.layers import Input

def trainModel(X, y):
    '''
        Training the Neural Network model against the data.
        Args: 
            X: Array of features to be trained.
            y: Array of Target attribute.

        Returns: Save Trained model weights.
    '''
	K.clear_session(X, y)
	batch_size=X.shape[0]
	time_steps=X.shape[1]
	data_dim=X.shape[2]

	Input_Sample = Input(shape=(time_steps,data_dim))
	Output_ = InstantiateModel(Input_Sample)
	Model_Enhancer = Model(inputs=Input_Sample, outputs=Output_)

	Model_Enhancer.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adamax())

	ES = EarlyStopping(monitor='val_loss', min_delta=0.5, patience=200, verbose=1, mode='auto', baseline=None,
                              restore_best_weights=False)
    MC = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='auto', verbose=0, save_best_only=True)
    
    #class_weights = class_weight.compute_sample_weight('balanced',
	#                                                 np.unique(y[:,0],axis=0),
	#                                                 y[:,0])
    ModelHistory = Model_Enhancer.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs,
                                  validation_data=(x_test, y_test),
                                  callbacks = [MC],
                                  #class_weight=class_weights,
                                  verbose=1)
