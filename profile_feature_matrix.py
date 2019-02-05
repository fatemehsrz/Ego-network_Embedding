import os
import random
import numpy as np


pathhack = os.path.dirname(os.path.realpath(__file__))
egoList=[0, 107, 1684, 1912 , 3437, 348, 3980, 414, 686, 698]

######################## reading ego's features and alters' features

egoFeat=[]
alterNodes=[]
alterFeatures=[]
for i in egoList:
    Nodes=[]
    Features=[]
    feat_file = ("%s/./../generate_local_embeddings/data/%d.egofeat"  % (pathhack,i))
    f1= open(feat_file, "r")
    for line in f1:
        line_file1= line.rstrip('\n').split(' ')
        egoFeat.append(line_file1) 


    egofeat_file = ("%s/data/%d.feat"  % (pathhack,i))
    
    f2= open(egofeat_file, "r")
    for line in f2:
        line_file2= line.rstrip('\n').split(' ')
          
        Nodes.append(line_file2[0])
        Features.append(line_file2[1:])
           
    alterNodes.append(Nodes)
    alterFeatures.append(Features) 
        
#######################################################   
 
egoFeatures=[]
for i in range(len(egoFeat)):
    a=[]
    for j in range(len(egoFeat[i])):
        a.append(int(egoFeat[i][j]))  
    egoFeatures.append(a)
    
    
LengthFeature=[]
for i in range(len(alterFeatures)):
    for j in range(len(alterFeatures[i])):
        
        print(len(alterFeatures[i][j]))
        LengthFeature.append(len(alterFeatures[i][j]))
        #print('featureLength:',LengthFeature)
        
featureLength=max(LengthFeature)

 
############ writing common features as vectors on the file

f3= open('common_feature.txt', 'w') 
 
for i in range(len(alterFeatures)): #10 egos
     
    for j in range(len(alterFeatures[i])): 
        
        c=[0]*featureLength
        for k in range(len(alterFeatures[i][j])): 
             
            if egoFeat[i][k]== alterFeatures[i][j][k] and egoFeat[i][k]== '1': # checking if ego and alter have a common feature
                print( 'in common feature matrix:','ego:', egoList[i], 'alter:',alterNodes[i][j], 'positin:', k, 'should be 1')
                c[k]=1
        print(len(c))
             
        f3.write(str(alterNodes[i][j])+' '+str(c)) 
        f3.write('\n')       
          
   
f3.close()   

#######################################################################################


# print(len(egoFeat))
# print(len(egoFeat[0]))
# print('len(egoFeat):',len(egoFeat))
# print('len(egoFeat 0):',egoFeat[0])
# print('len(alterNodes):',len(alterNodes))
# print('len(alterNodes[0]):',alterNodes[0])
# print('len(alterFeatures):',len(alterFeatures))
# print('len(alterFeatures [0]):',len(alterFeatures[0]))
# print('len(alterFeaturess[0][0] ):',len(alterFeatures[0][0]))
# print('len(alterFeaturess[0][0][0] ):',len(alterFeatures[0][0][0]))



