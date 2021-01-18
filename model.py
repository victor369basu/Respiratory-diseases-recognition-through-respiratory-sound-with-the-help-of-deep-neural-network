from keras.utils import np_utils
from keras.layers import add, Conv2D,Input,BatchNormalization,TimeDistributed,Embedding,LSTM,GRU,Dense,MaxPooling1D,Dropout,LeakyReLU,ReLU,Flatten,concatenate,Bidirectional
from keras.layers.merge import concatenate
from keras.models import Model,load_model

def InstantiateModel(in_):
   '''
      Architecture of the Deep Learning Model.
      Args:
        in_: input tensor shape
      Returns: Tensor model
   '''
    model_2_1 = GRU(32,return_sequences=True,activation=None,go_backwards=True)(in_)
    model_2 = LeakyReLU()(model_2_1)
    model_2 = GRU(128,return_sequences=True, activation=None,go_backwards=True)(model_2)
    #model_2 = BatchNormalization()(model_2)
    model_2 = LeakyReLU()(model_2)
    
    model_3 = GRU(64,return_sequences=True,activation=None,go_backwards=True)(in_)
    model_3 = LeakyReLU()(model_3)
    model_3 = GRU(128,return_sequences=True, activation=None,go_backwards=True)(model_3)
    #model_3 = BatchNormalization()(model_3)
    model_3 = LeakyReLU()(model_3)
    
    model_add_1 = add([model_3,model_2])
    
    model_5 = GRU(128,return_sequences=True,activation=None,go_backwards=True)(model_add_1)
    model_5 = LeakyReLU()(model_5)
    model_5 = GRU(32,return_sequences=True, activation=None,go_backwards=True)(model_5)
    model_5 = LeakyReLU()(model_5)
    
    model_6 = GRU(64,return_sequences=True,activation=None,go_backwards=True)(model_add_1)
    model_6 = LeakyReLU()(model_6)
    model_6 = GRU(32,return_sequences=True, activation=None,go_backwards=True)(model_6)
    model_6 = LeakyReLU()(model_6)
    
    model_add_2 = add([model_5,model_6,model_2_1])
    
    
    model_7 = Dense(64, activation=None)(model_add_2)
    model_7 = LeakyReLU()(model_7)
    model_7 = Dropout(0.2)(model_7)
    model_7 = Dense(16, activation=None)(model_7)
    model_7 = LeakyReLU()(model_7)
    
    model_9 = Dense(32, activation=None)(model_add_2)
    model_9 = LeakyReLU()(model_9)
    model_9 = Dropout(0.2)(model_9)
    model_9 = Dense(16, activation=None)(model_9)
    model_9 = LeakyReLU()(model_9)
    
    model_add_3 = add([model_7,model_9])

    model_10 = Dense(16, activation=None)(model_add_3)
    #model_10 = BatchNormalization()(model_10)
    model_10 = LeakyReLU()(model_10)
    model_10 = Dropout(0.5)(model_10)
    #Model_7 = MaxPooling1D(pool_size=2)(mode)
    model_10 = Dense(6, activation="softmax")(model_10)
    
    return model_10