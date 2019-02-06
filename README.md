# Introduction

An ego-network organizes the social relationships between an individual (ego) and others
(alters) into different groups (social circles). 

# Paper
[Global and Local Feature Learning for Ego-Network Analysis](https://www.uni-weimar.de/medien/webis/events/tir-17/tir17-papers-final/SalehiRizi2017_global-and-local-features-learning-for-ego-network-analysis_paper.pdf)

# Datasets
[Facebook](https://snap.stanford.edu/data/egonets-Facebook.html) <br />
[Twitter](https://snap.stanford.edu/data/egonets-Twitter.html) <br />
[Google+](https://snap.stanford.edu/data/egonets-Gplus.html) <br />


In this study, we want to predict social circle for a new alter added to the ego-network.<br />

1) Each alter belongs to one or more circles. Therefore, we need a multi-label classifier. We use a Feed Forward NN classifier.<br />

2) Our Feed Forward Neural Network classifier with three following inputs:<br />
   - loc(u)+glo(v) <br />
   - glo(u)+glo(v) <br />
   - loc(u)+glo(u)+glo(v) <br />

   - loc(u)+glo(v)+sim(u,v) <br />
   - glo(u)+glo(v)+sim(u,v) <br />
   - loc(u)+glo(u)+glo(v)+sim(u,v)<br />

# global and local vectors

glo(v): applying DeepWalk to the Facebook garph returns global vectors where embedding size= 300 and windowsize=2.<br />
loc(u): applying ParagraphVector to ego-walks (walks over one ego-network) returns local vector for each ego where embedding size= 300 and windowsize=2.<br />

# Files and folders
 
| file or folder                  |                                           function or content                                                  | 
| ------------------------------- |--------------------------------------------------------------------------------------------------------------- |
|binaryLabels.csv                 | alters' lables in the binary form                                                                              |
|common_features.txt              | common profile features between ego u and alter v                                                              |
|locglo.py                        | applying NN classifier where each row of feature matrix is loc(u)+glo(v)                                       | 
|gloglo.py                        | applying NN classifier where each row of feature matrix is glo(u)+glo(v)                                       |
|locgloglo.py                     | applying NN classifier where each row of feature matrix is loc(u)+glo(u)+glo(v)                                |
|locglo_sim.py                    | applying NN classifier where each row of feature matrix is loc(u)+glo(v)+sim(u,v)                              | 
|gloglo_sim.py                    | applying NN classifier where each row of feature matrix is glo(u)+glo(v)+sim(u,v)                              |
|locgloglo_sim.py                 | applying NN classifier where each row of feature matrix is loc(u)+glo(u)+glo(v)+sim(u,v)                       |







 
