import time
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import np_utils
import keras.callbacks as cb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.datasets import mnist
from gensim.models import Word2Vec
from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
import pandas
import csv
import os
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support
import scipy.sparse as sp
from sklearn.utils import shuffle as skshuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict
from keras.callbacks import CSVLogger

class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)


def init_model():
    start_time = time.time()
    print ('Compiling Model ... ')
    
    model = Sequential()
    model.add(Dense(900, input_dim=900)) # 300+300+300
    model.add(Activation('relu'))
    
    model.add(Dropout(0.4))
    
    model.add(Dense(600))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.4))
    
    model.add(Dense(46))                 # 46 circles
    model.add(Activation('softmax'))

    rms = RMSprop()
    
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['fbeta_score', 'precision', 'recall'])
    
    
    return model


def run_network(data=None, model=None, epochs=20, batch= 32):
    try:
        start_time = time.time()
        if data is None:
            X_train, X_test, y_train, y_test = load_data()
        else:
            X_train, X_test, y_train, y_test = data

        if model is None:
            model = init_model()

        #history = LossHistory()

        print( 'Training model...')
        
        csv_logger = CSVLogger('locgloglo.log')
        model.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch,
                  callbacks=[csv_logger] , validation_split= 0.3, verbose=2)
        
        score = model.evaluate(X_test, y_test, batch_size=32)
        print( "Network's test score [loss, fbeta_score, precision, recall]: {0}".format(score))
        
    except KeyboardInterrupt:
        print( ' KeyboardInterrupt')
        
    return score

            
#################### Load Embeddings

embeddings_file1 = "./../generate_global_embeddings/glo300_win3.emb"
model1 = Word2Vec.load_word2vec_format(embeddings_file1, binary=False)

embeddings_file2 = open('./../generate_local_embeddings/ego_local.emb', "r")

egoVector = []
for line in embeddings_file2:
   a1= line.rstrip('\n').split(' ') [1:]
   a2= [float(i) for i in a1]
   egoVector.append(a2)


######################## Load labels

dataframe = pandas.read_csv("binaryLabels.csv", delimiter= ' ',header=None)
nodeLabel = dataframe.values

node = nodeLabel[:,0]
binLabel = nodeLabel[:,1:47]
#print(binLabel[0])
labels_matrix= np.array(binLabel)
print('labels_matrix.shape:',labels_matrix.shape)

 
####################### Load features 

egoList=[0, 107, 1684, 1912 , 3437, 348, 3980, 414, 686, 698]
pathhack = os.path.dirname(os.path.realpath(__file__))

circlesList=[]
for i in egoList:
    
    circle_file = ("%s/./../Facebook_data/data/%d.circles"  % (pathhack,i))
    with open(circle_file, "r") as f1:
        circleNodes = [line.rstrip('\n').split('\t') for line in f1]
        circlesList.append(circleNodes)
        
egoCircleNodes=[]    
for i in range(len(egoList)):
    a=[]
    for j in range(len(circlesList[i])):
         a.extend(circlesList[i][j][1:])
    egoCircleNodes.append(a)  
     


############################### Vector concatenation local ego and global alters
             
EgoAlterVectors=[]
for i in range(len(egoList)):       
    for j in egoCircleNodes[i]:
        
        
        #print('i:',i, 'j:',j)
        con1=np.concatenate((egoVector[i], model1[str(egoList[i])]), axis=0) # loc glo glo sim
        con2=np.concatenate((con1, model1[str(j)]), axis=0) # 
        
        EgoAlterVectors.append(con2)
                      
features_matrix = np.array(EgoAlterVectors)

print('features_matrix.shape:',features_matrix.shape)


########################## Train and Test sets

X=features_matrix
y= labels_matrix


sss = StratifiedShuffleSplit(n_splits=4, test_size=0.3, random_state=0)
sss.get_n_splits(X, y)

Micro_values = defaultdict(list)
Macro_values = defaultdict(list)
fscoreMicro=[]
fscoreMacro=[]    
fscoreMain =[]   

##################### f-score    

for train_index, test_index in sss.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index] 

         
    data1=  X_train, X_test, y_train, y_test
       
    fscore1= run_network(data=data1, epochs=50, batch= 32)
    
    fscoreMain.append(fscore1[1]) 
    
    

print(fscoreMain)  
print('loc+glo+glo classifier F1 score: ',np.mean(fscoreMain)) 


   
    
        
f=open('locgloglo.fscore', 'w')
      
f.write(str( fscoreMain))
f.write('\n')
f.write('average F1 score:'+ str(np.mean(fscoreMain)))
f.write('\n')
  
f.close()  
  
             


