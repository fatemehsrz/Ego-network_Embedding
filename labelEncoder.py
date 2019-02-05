import sklearn
import pandas
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils



dataframe = pandas.read_csv("node_circle.csv", delimiter= ' ',header=None)
labelset = dataframe.values
y1 = labelset[:,0] #node
y2 = labelset[:,1] #circle
y3=[]
  
f3=open('node_circle2.txt', 'w')
for i in range(len(y1)):
     
     id=int(y2[i][6:])
     y3.append(id)
     print('node:',y1[i], 'circle:',int(y3[i]))
     f3.write(str(y1[i])+' '+str(y3[i]))
     f3.write("\n")
     #print(y3)
f3.close
     
from collections import defaultdict
 
def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() 
                            if len(locs)>1)
duplicate1= dict()
for dup in sorted(list_duplicates(y1)):
    duplicate1.update({dup[0]:dup[1]})
   

print(duplicate1)
print(list(duplicate1))
print('-----------------------------------------------')
multiLabels=dict()
  
f1= open('multiLabels.txt','w')
y4= []
for i in range(len(y1)):
    a=[]
    if y1[i] in list(duplicate1):
        for j in range(len(duplicate1[y1[i]])):
            d= duplicate1[y1[i]][j]
            a.append(y2[d])
        print('node:', y1[i], 'label:', a )   
        multiLabels.update({y1[i]:a})
        node1= {y1[i]: a} 
        f1.write(str(node1))
        f1.write("\n")
    else:
       print('node:', y1[i], 'label:', [y2[i]] )  
       multiLabels.update({y1[i]:[y2[i]]}) 
       node2= {y1[i]:[y2[i]]}
       f1.write(str(node2))
       f1.write("\n")
      
print(multiLabels)
  
  
lb = preprocessing.MultiLabelBinarizer()
mLabel=[]
for i in y1:
    mLabel.append(multiLabels[i])
    print(i,'  ',multiLabels[i])
      
mm=lb.fit_transform(mLabel)  
f2= open('multiLabelBinarized.txt','w') 
for i in range(len(mm)):
    f2.write(str(y1[i])+' ')
    for j in range(len(mm[i])):
        f2.write(str(mm[i][j])+' ')
    f2.write("\n")
f2.close
  
  
  
print(lb.classes_)


