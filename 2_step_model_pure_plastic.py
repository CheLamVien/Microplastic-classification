# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:53:49 2021

@author: Vien Che
"""


import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
import itertools
import pandas as pd
import time
import joblib
from matplotlib import rcParams
from sklearn.decomposition import PCA 
from sklearn.metrics import precision_recall_curve,precision_score,recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from matplotlib.patches import Circle
#####################
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize= 23)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,fontsize= 18)
    plt.yticks(tick_marks, classes,fontsize= 18)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',fontsize= 18)
    plt.xlabel('Predicted label',fontsize= 18)
#########
def unique(list1): 
  
    # intilize a null list 
    unique_list = [] 
      
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x)
    return unique_list
##################
def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique),loc='upper center',bbox_to_anchor=(0.5, 1.1),
         fancybox=True, shadow=True, ncol=8,fontsize=12)
################


plastic_name_1 =['PE_BAM','PE_BPI','PE300_VG','PE500_VG','PE1000_VG']
plastic_name_2 =['PVC_BAM','PVC_BPI','PVC_VG']
plastic_name_3=['LDPE_BAM','LDPE_BPI','LDPE_Brett','HDPE_Brett']
plastic_name_4=['PP_BAM','PP_BPI','PP_Brett','PP_VG']
plastic_name_5=['PA_BAM','PA_BPI','PA_Brett','PA_VG']
plastic_name_6=['PC_BAM','PC_BPI','PC_Brett','PC_VG']




plastic_name_7 =['PET_BAM','PET_BPI','PET_Brett','PET_VG']
plastic_name_8 =['PMMA_BAM','PMMA_BPI','PMMA_Brett','PMMA_VG']
plastic_name_9=['PS_BAM','PS_BPI','PS_Brett','PS_VG']
nonpl_name_1=['N1_nature','N2_nature','N3_nature','N4_nature']
nonpl_name_2=['N5inner_nature','N5outer_nature','N6inner_nature','N6outer_nature']
nonpl_name_3=['N7_nature','N8_nature','N9_nature','N10_nature']


    
    
sample_name=plastic_name_1+plastic_name_2+plastic_name_3+plastic_name_4+plastic_name_5+plastic_name_6+plastic_name_7+plastic_name_8+plastic_name_9+nonpl_name_1+nonpl_name_2+nonpl_name_3

sample_name_taget=['PE','PE','PE','PE','PE',
                   'PVC','PVC','PVC',
                   'PE','PE','PE','PE',
                   'PP','PP','PP','PP',
                   'PA','PA','PA','PA',
                   'PC','PC','PC','PC',
                    'PET','PET','PET','PET',
                   'PMMA','PMMA','PMMA','PMMA',
                   'PS','PS','PS','PS',
                   'N1','N2','N3','N4',
                   'N5','N5','N6','N6',
                   'N7','N8','N9','N10'
                   ]

sample_labels=['PVC','PMMA','PS','PP','PET','PE','PC','PA','Natural material']

#convert plastic types (categories) into number
sample_labels1=['Plastic','Non Plastic']
df1=pd.DataFrame(sample_labels1,columns={'sample_name'}) #change the type of the column
#convert categories to numbers
df1.sample_name = pd.Categorical(df1.sample_name)
df1['code'] = df1.sample_name.cat.codes
print(df1)


df=pd.DataFrame(sample_labels,columns={'sample_name'}) #change the type of the column
df.sample_name = pd.Categorical(df.sample_name)
df['code'] = df.sample_name.cat.codes
print(df)



############################ Design matrix ##########
dataf = []

for i in range(len(sample_name)):
    data=np.load('E:/destop/1st measurement PL/Spectra/norm/'+sample_name[i]+'.npy')
    if sample_name[i]== 'N5inner_nature' or sample_name[i]== 'N5outer_nature' or sample_name[i]== 'N6inner_nature' or sample_name[i]== 'N6outer_nature':
        n=10       #number of spectra you want in the data array
        data=data[0:n,:]
    else:
        if sample_name[i]== 'N1_nature' :
            n=21       #number of spectra you want in the data array
            data=data[0:n,:]
            data=np.delete(data, (5), axis=0) #delete uncommon spectra of N1 sample
        if sample_name[i]== 'PC_VG':
            n=21       #number of spectra you want in the data array
            data=data[0:n,:]
            data=np.delete(data, (11), axis=0) #delete uncommon spectra of PC_VG sample
        if sample_name[i]== 'LDPE_BAM':
            n=21       #number of spectra you want in the data array
            data=data[0:n,:]
            data=np.delete(data, (0), axis=0) #delete uncommon spectra of PC_VG sample
        else:
            n=20      #number of spectra you want in the data array
            data=data[0:n,:]
    dataf.append(data)
    print(sample_name[i]+' has ', data.shape[0], 'spectra' )

#X_list is a design matrix which is a flatten list of dataf
# each spetra has an intensity from 410 to 800 nm
X_list= np.array([item for sublist in dataf for item in sublist])
print('the total number of spectra is ', X_list.shape[0])


#############SMOOTHEN##########
#X_list= scipy.signal.savgol_filter(X_list, 11, 2) # window size 51, polynomial order 3

#########Target vector for plastic and non-plastic classification #########

sample_name_taget_plastics=['Plastic' for i in range(0,36)] #36 types of plastic from 3 companies
sample_name_taget_non_plastics=['Non_Plastic' for i in range(0,12)] #12 types of non plastic
sample_name_taget1=sample_name_taget_plastics+sample_name_taget_non_plastics



Y_list1= [] #empty list for defining plastic or non-plastic
Y_list2= [] #empty list for defining sample names with the company
Y_list3= [] #empty list for defining plastic types regardless of the company
for i in range(len(sample_name_taget1)):
    name=[sample_name_taget1[i] for k in range(0,len(dataf[i]))]
    name1=[sample_name[i] for k in range(0,len(dataf[i]))]
    name2=[sample_name_taget[i] for k in range(0,len(dataf[i]))]
    Y_list1.append(name)
    Y_list2.append(name1)
    Y_list3.append(name2)

flattend_Y_list1= [item for sublist in Y_list1 for item in sublist]
flattend_Y_list2= [item for sublist in Y_list2 for item in sublist]
flattend_Y_list3= [item for sublist in Y_list3 for item in sublist]
df_y1=pd.DataFrame({'sample_type_1':flattend_Y_list1,'sample_type_2':flattend_Y_list3,'sample_name':flattend_Y_list2}) 
df_y1['sample_type_1'] = pd.Categorical(df_y1.sample_type_1)#change the type of the column
df_y1['sample_type_2'] = pd.Categorical(df_y1.sample_type_2)
#convert categories to numbers
df_y1['code'] = df_y1.sample_type_1.cat.codes #code for plastic and non plastic classification
df_y1['code_1'] = df_y1.sample_type_2.cat.codes #code for plastic type classification

Y=np.array(df_y1[['code','code_1']])


####################################split data #############
X_train, X_test , y_train_split, y_test_split = train_test_split( X_list, Y, 
                                                            test_size=0.2, random_state=4)
#Sklearn train_test_split function ignores the original sequence of numbers. After a split, they can be presented in a different order




y_train=y_train_split[:,0] #target vector for plastic vs nature material classification
y_test=y_test_split[:,0]

'''
#change the shape of y to (n_samples, ) to suitable with LR model in Sklearn
Y1=np.ravel(Y1) 
Y2=np.ravel(Y2)
'''
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)



################### First pipeline ########
pca=PCA(0.95) #no. of PCs explained 95% of variance
LRCV=LogisticRegressionCV(solver='lbfgs',penalty ='l2',multi_class='ovr',
                              max_iter=5000,random_state=4,n_jobs=2)
model_1 = Pipeline([("PCA", pca),("LRCV",LRCV )]) # first apply PCA and then train model using LRCV



################### CROSS VALIDATION ##########################
def cross_validation (model):    
    scores = cross_validate(model , X_train, y_train, cv=10,scoring=('accuracy', 'precision_macro','recall_macro'),
                        return_train_score=True)

    train_accuracy=scores['train_accuracy']
    print('Train accuracy: ',round(np.mean(train_accuracy)*100,2),'% +/-', round(np.std(train_accuracy)*100,2))
    test_accuracy=scores['test_accuracy']
    print('Validation accuracy: ',round(np.mean(test_accuracy)*100,2),'% +/-', round(np.std(test_accuracy)*100,2))
    train_pre=scores['train_precision_macro']
    print('Train precision: ',round(np.mean(train_pre),2),'% +/-', round(np.std(train_pre),2))
    test_pre=scores['test_recall_macro']
    print('Validation precision: ',round(np.mean(test_pre),2),'% +/-', round(np.std(test_pre),2))
    train_rec=scores['train_precision_macro']
    print('Train recall: ',round(np.mean(train_rec),2),'% +/-', round(np.std(train_rec),2))
    test_rec=scores['test_recall_macro']
    print('Validation recall: ',round(np.mean(test_rec),2),'% +/-', round(np.std(test_rec),2))

cross_validation(model_1)
#################### Final training with full training set and evaluation on test set ########

def final_training (model):
    start_time = time.time()

    model.fit(X_train,y_train)
    acc_train= model.score(X_train, y_train)
    acc_test= model.score(X_test, y_test)
    print ('Training accuracy:', round((acc_train)*100,2))
    print ('Test accuracy:', round((acc_test)*100,2))
    recal_train=recall_score(y_train,model.predict(X_train))
    recal_test=recall_score(y_test,model.predict(X_test))
    print ('Training recall:', round((recal_train),2))
    print ('Test recall:', round((recal_test),2))
    pre_train=precision_score(y_train,model.predict(X_train))
    pre_test=precision_score(y_test,model.predict(X_test))
    print ('Training pre:', round((pre_train),2))
    print ('Test pre:', round((pre_test),2))
    t=(time.time() - start_time)/60
    print ("My program took", t , "to run (mins)")

final_training (model_1)
# save the model to disk
filename = 'E:/destop/1st measurement PL/model_1.sav'
# Save the model as a pickle in a file 
joblib.dump(model_1, filename)

#Check how many PCs explained 95% of variance 
print('no. of PCs explained 95% of variance', pca.components_.shape[0])









######### PLOT DECISION BOUNDARY ################
model_1= joblib.load(filename)
import seaborn as sns
sns.set(style="white")
pca.fit(X_train)  
X_list_pca=pca.transform(X_list)  
explained_variance = pca.explained_variance_ratio_    
principalDf = pd.DataFrame(data = X_list_pca
             , columns = ['principal component 1', 'principal component 2','principal component 3'])

finalDf = pd.concat([principalDf, df_y1], axis = 1)
#########plot withot company###########
fig = plt.figure(figsize = (10,15))
ax = fig.add_subplot(111, projection='3d')

#ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC 1', fontsize = 12)
ax.set_ylabel('PC 2', fontsize = 12)
ax.set_zlabel('PC 3', fontsize = 12)
#ax.set_title('2 component PCA', fontsize = 20)
targets = ['Plastic','Non_Plastic']
colors = ['black', 'red']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['sample_type_1'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , finalDf.loc[indicesToKeep, 'principal component 3']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

# The equation of the separating plane is given by all x so that np.dot(LRCV.coef_[0], x) + b = 0.
# Solve for w3 (z)
z = lambda x,y: (-LRCV.intercept_[0]-LRCV.coef_[0][0]*x -LRCV.coef_[0][1]*y) / LRCV.coef_[0][2]
xx, yy = np.mgrid[-0.05:0.6:.01, -0.05:0.6:.01]
#xx, yy = np.mgrid[-5:5:.01, -5:5:.01]
zz=z(xx,yy)   
# plot the surface
ax.plot_surface(xx, yy, zz, color='blue',alpha=0.5)
ax.view_init(30, 60)
#ax.set_zlim(-.05,0.2)

######### PLOT WITH DETAILED LABLED #########
### IMAGE GET FROZEN WHEN ZOOMING IN###### NEED MORE POWER COMPUTER########
##plot with company ###########
import matplotlib.cm as cm
import matplotlib.colors as colors
colormap2 = cm.tab20b
color_list = [colors.rgb2hex(colormap2(i)) for i in np.linspace(0, 0.9, 10)] # 10 is no. of non-plastic sample


fig = plt.figure(figsize = (10,15))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('PC 1', fontsize = 12)
ax.set_ylabel('PC 2', fontsize = 12)
ax.set_zlabel('PC 3', fontsize = 12)

targets = sample_name
    
for i in range (len(Y)):
    if flattend_Y_list2[i][-1]== 'M':
        marker='o'
    elif flattend_Y_list2[i][-1]== 'I':
        marker='*'
    elif flattend_Y_list2[i][-1]== 'G':
        marker='s'
    elif flattend_Y_list2[i][-1]== 't':
        marker='v'
    else: 
        marker='>'
    if flattend_Y_list3[i]=='PE' or flattend_Y_list3[i] =='N1':
        c='red'
    if flattend_Y_list3[i]=='PVC' or flattend_Y_list3[i] =='N2':
        c='black'
    if flattend_Y_list3[i] == 'PS' or flattend_Y_list3[i] =='N3':
        c='green'
    if flattend_Y_list3[i] =='PMMA' or flattend_Y_list3[i] == 'N4':
        c='blue'
    if flattend_Y_list3[i] == 'PP' or flattend_Y_list3[i] == 'N5':
        c='orange'
    if flattend_Y_list3[i] == 'PET' or flattend_Y_list3[i] == 'N6':
        c='yellow'
    if flattend_Y_list3[i] == 'PC' or flattend_Y_list3[i] == 'N7':
        c='pink'
    if flattend_Y_list3[i] == 'PA' or flattend_Y_list3[i] == 'N8':
        c='lightslategrey'
    if flattend_Y_list3[i] == 'N9':
        c='purple'
    if flattend_Y_list3[i] == 'N10':
        c='turquoise'
    try:
        ax.scatter(finalDf.loc[i, 'principal component 1']
                   , finalDf.loc[i, 'principal component 2']
                   , finalDf.loc[i, 'principal component 3']
                   , c = c
                   , s = 50
                   , marker= marker
                   ,label=flattend_Y_list2[i])
    except:
        pass
#legend_without_duplicate_labels(ax)
ax.plot_surface(xx, yy, zz, color='blue',alpha=0.5)
ax.view_init(30, 60)
ax.grid()

################### Compute confusion matrix #############################3

loaded_model= joblib.load(filename)
yhat=loaded_model.predict(X_test) #the original recall
labels=np.ravel(np.array(df1[['code']] ))  
cnf_matrix = confusion_matrix(y_test, yhat, labels=labels)
np.set_printoptions(precision=2)

plt.figure(figsize=(10,8))
plot_confusion_matrix(cnf_matrix, classes=sample_labels1,normalize= False,  title='Confusion matrix')

############### Check wrong spectra #################3
taxis=np.load('E:/spectra/sample/taxis.npy') #x_axis   
  
#plot the wrong prediction spectrum

X=X_test
y=y_test
'''
X=X_train
y=y_train
'''

yhat=loaded_model.predict(X)
find_index=np.where(yhat != y) #find index of wrong prediction
find_index=np.ravel(find_index)
print('the number of wrong prediction: ',len(find_index))
name_wrong=[]
plt.figure(figsize=(10,8))  
for i in find_index: 
    norm=X[i]
    index_in_x_list=int(np.ravel(np.where(np.all(norm==X_list,axis=1))))
    name=df_y1.iloc[index_in_x_list]['sample_name']
    name_wrong.append(name)
    plt.plot(taxis[658:2681],norm,label=name,linewidth=2) #plot all spectra of each sample with non nomalization
    plt.legend() 
    plt.grid(b=True, which='major', color='k', linestyle='-', alpha=0.2)
    plt.grid(b=True, which='minor', color='b', linestyle='--', alpha=0.1)
    plt.minorticks_on()
    plt.xlabel('Wavelength (nm)',fontsize=15)
    plt.xticks(fontsize=13)
    plt.ylabel('Intensity (arb.u.)',fontsize=15)
    plt.yticks(fontsize=13)

    leg = plt.legend(loc=1, numpoints=1)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)

from collections import Counter
print(Counter(name_wrong).keys()) # equals to list(set(words))
print(Counter(name_wrong).values()) # counts the elements' frequency


########## make prediction on mixed sample #########
filename = 'E:/destop/1st measurement PL/model_2.sav'
loaded_model= joblib.load(filename)
########################## loading mixed sample data#############################
img_file=  cbook.get_sample_data('E:/destop/1st measurement PL/c.jpg')
img=plt.imread(img_file)
mixed_data=np.load('E:/destop/1st measurement PL/mixed_spectra.npy') #loading spectra
x=np.load('E:/destop/1st measurement PL/x.npy') #x coordinates
y=np.load('E:/destop/1st measurement PL/y.npy') #y coordinates

   
yhat = loaded_model.predict(mixed_data)
y_true= np.load('E:/destop/1st measurement PL/y_true_pl_vs_non_pl.npy')
find_index_mixed=np.where(yhat != y_true) #find index of wrong prediction
find_index_mixed=np.ravel(find_index_mixed)
print('the number of wrong prediction: ',len(find_index_mixed))
print('accuracy', (len(y_true)-len(find_index_mixed))/len(y_true))
##############plot result##############
colors = ['b','r']
color_for_mapping=[]
label_for_mapping=[]

for i in range (len(yhat)):
    for j in range(0,2):
        if yhat[i] == df1['code'][j]:
            label=df1['sample_name'][j]
            color=colors[j]
            
    color_for_mapping.append(color)
    label_for_mapping.append(label)


# Create a figure. Equal aspect so circles look circular
fig,ax = plt.subplots(figsize=(15,10))
ax.set_aspect('equal')
ax.set_axis_off()
fig.add_axes(ax)
# Show the image
ax.imshow(img) 

# Now, loop through coord arrays, and create a circle at each x,y pair
for xx,yy,aa,bb in zip(x,y,color_for_mapping,label_for_mapping):
    circ = Circle((xx,yy),20,color=aa,label=bb)
    ax.add_patch(circ)
# Put a legend below current axis
legend_without_duplicate_labels(ax)
# Show the image
plt.show()




########session to plot explained variance#########
def plot_ex_variance (data):
    pca = PCA(n_components = 10)
    if data == 'non_scaled':
            pca.fit(X_train) #non_scaled
    if data == 'scaled':
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled=scaler.transform(X_train)
        pca.fit(X_train_scaled)
    explained_variance = pca.explained_variance_ratio_ *100
    total=sum(explained_variance)
    x=['PC'+str(i) for i in range(1,11)]
    print(total) 
    plt.figure(figsize=(10,20))
    plt.bar(x,explained_variance,color='black')
    plt.ylabel('Percentage of explained variance (%)',fontsize=15)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.ylim(0,70)

plot_ex_variance(data= 'non_scaled')
plot_ex_variance(data= 'scaled')


################### CALCULATE RECALL AND PRECISION##########################
################### RUN THIS PART OF CODES TO INCREASE RECALL ########################
from sklearn.model_selection import cross_val_predict
y_scores_train = cross_val_predict(loaded_model, X_train, y_train, cv=3, method="decision_function")
y_scores_test = loaded_model.decision_function(X_test)
            #######PLOT RECALL PRECISION RELATIonSHIP#########

precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores_train)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(recalls[:-1], precisions[:-1])
    plt.grid(b=True, which='major', color='k', linestyle='-', alpha=0.2)
    plt.grid(b=True, which='minor', color='b', linestyle='--', alpha=0.1)
    plt.minorticks_on()
    plt.xlabel('Recall',fontsize=rcParams['axes.labelsize'])
    plt.xticks(fontsize=rcParams['axes.labelsize'])
    plt.ylabel('Precision',fontsize=rcParams['axes.labelsize'])
    plt.yticks(fontsize=rcParams['axes.labelsize'])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
        ################## aim for 100% precision###########
threshold_100_precision = thresholds[np.argmax(recalls >= 1)]
threshold_95_precision = thresholds[np.argmax(recalls <= .951)]
threshold_97_precision = thresholds[np.argmax(recalls <= .971)]
threshold_99_precision = thresholds[np.argmax(recalls <= .991)]
        ################## make predictions ##############
threshold=threshold_95_precision
y_train_pred_100 = (y_scores_train >= threshold)
y_test_pred_100=(y_scores_test >= threshold)

p_tr=precision_score(y_train, y_train_pred_100)
r_tr=recall_score(y_train, y_train_pred_100)
a_tr=100-(len(np.ravel(np.where(y_train!=y_train_pred_100)))/len(y_train) *100)
print(p_tr,r_tr,a_tr)

p_t=precision_score(y_test, y_test_pred_100)
r_t=recall_score(y_test, y_test_pred_100)
a_t=100-(len(np.ravel(np.where(y_test!=y_test_pred_100)))/len(y_test) *100)
print(p_t,r_t,a_t)











################# second pipeline with standardizing the features #################
pca=PCA(0.95) #no. of PCs explained 95% of variance
LRCV=LogisticRegressionCV(solver='lbfgs',penalty ='l2',multi_class='ovr',
                              max_iter=5000,random_state=4,n_jobs=2)
scaler = StandardScaler()
model_2 = Pipeline([("scaled",scaler),("PCA", pca),("LRCV",LRCV )])
###### cross_validation #########
cross_validation(model_2)
####### final training ###
final_training(model_2)
# save the model to disk
filename = 'E:/destop/1st measurement PL/model_2.sav'
# Save the model as a pickle in a file 
joblib.dump(model_2, filename)

#Check how many PCs explained 95% of variance 
print('no. of PCs explained 95% of variance', pca.components_.shape[0])
###################### PLOT WRONG SPECTRA AND PVC_BAM SPECTRA ###########
taxis=np.load('E:/spectra/sample/taxis.npy') #x_axis   

X=X_test
y=y_test
yhat=loaded_model.predict(X)
find_index=np.where(yhat != y) #find index of wrong prediction
find_index=np.ravel(find_index)
plt.figure(figsize=(10,8))  
for i in find_index: 
    norm=X[i]
    index_in_x_list=int(np.ravel(np.where(np.all(norm==X_list,axis=1))))
    name=df_y1.iloc[index_in_x_list]['sample_name']
    name_wrong.append(name)
    plt.plot(taxis[658:2681],norm,label=name) #plot all spectra of each sample with non nomalization
    plt.legend() 
    plt.grid(b=True, which='major', color='k', linestyle='-', alpha=0.2)
    plt.grid(b=True, which='minor', color='b', linestyle='--', alpha=0.1)
    plt.minorticks_on()
    plt.xlabel('Wavelength (nm)',fontsize=15)
    plt.xticks(fontsize=13)
    plt.ylabel('Intensity (arb.u.)',fontsize=15)
    plt.yticks(fontsize=13)

    leg = plt.legend(loc=1, numpoints=1)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
ref_predicted1= np.load('E:/destop/1st measurement PL/Spectra/norm/PVC_BAM.npy')                
mean_pl1=np.mean(ref_predicted1,0)
std_pl1 = np.std(ref_predicted1,0)
plt.plot(taxis[658:2681], mean_pl1,linewidth=0.5,label="PVC_BAM",color='red')   
plt.fill_between(taxis[658:2681], mean_pl1-std_pl1, mean_pl1+std_pl1, alpha=0.2,color='red')
plt.legend()


















############################ TRAINING PLASTIC TYPE CLASSIFIER ################33
sample_name=plastic_name_1+plastic_name_2+plastic_name_3+plastic_name_4+plastic_name_5+plastic_name_6+plastic_name_7+plastic_name_8+plastic_name_9


############################ Design matrix ##########
dataf = []

for i in range(len(sample_name)):
    data=np.load('E:/destop/1st measurement PL/Spectra/norm/'+sample_name[i]+'.npy')
    if sample_name[i]== 'PC_VG':
        #print('hehe')
        n=21       #number of spectra you want in the data array
        data=data[0:n,:]
        data=np.delete(data, (11), axis=0) #delete uncommon spectra of PC_VG sample
    if sample_name[i]== 'LDPE_BAM':
        #print('hihi')
        n=21       #number of spectra you want in the data array
        data=data[0:n,:]
        data=np.delete(data, (0), axis=0) #delete uncommon spectra of PC_VG sample
    else:
        #print('huhu')
        n=20      #number of spectra you want in the data array
        data=data[0:n,:]
    dataf.append(data)
    print(sample_name[i]+' has ', data.shape[0], 'spectra' )

#X_list is a design matrix which is a flatten list of dataf
# each spetra has an intensity from 410 to 800 nm
X_list= np.array([item for sublist in dataf for item in sublist])
print('the total number of spectra is ', X_list.shape[0])

sample_name_taget=['PE','PE','PE','PE','PE',
                   'PVC','PVC','PVC',
                   'PE','PE','PE','PE',
                   'PP','PP','PP','PP',
                   'PA','PA','PA','PA',
                   'PC','PC','PC','PC',
                    'PET','PET','PET','PET',
                   'PMMA','PMMA','PMMA','PMMA',
                   'PS','PS','PS','PS'
                   ]
# sample names regardless their manufacturers 
sample_labels=['PVC','PMMA','PS','PP','PET','PE','PC','PA']

#convert plastic types (categories) into number
df1=pd.DataFrame(sample_labels,columns={'sample_name'}) #change the type of the column
#convert categories to numbers
df1.sample_name = pd.Categorical(df1.sample_name)
df1['code'] = df1.sample_name.cat.codes
print(df1)

#########Target vector for plastic classification #########

Y_list2= [] #empty list for defining sample names with the company
Y_list3= [] #empty list for defining plastic types regardless of the company
for i in range(len(sample_name_taget)):
    name1=[sample_name[i] for k in range(0,len(dataf[i]))]
    name2=[sample_name_taget[i] for k in range(0,len(dataf[i]))]
    Y_list2.append(name1)
    Y_list3.append(name2)

flattend_Y_list2= [item for sublist in Y_list2 for item in sublist]
flattend_Y_list3= [item for sublist in Y_list3 for item in sublist]
df_y1=pd.DataFrame({'sample_type':flattend_Y_list3,'sample_name':flattend_Y_list2}) 
df_y1['sample_type'] = pd.Categorical(df_y1.sample_type)#change the type of the column
#convert categories to numbers
df_y1['code'] = df_y1.sample_type.cat.codes 

Y=np.array(df_y1[['code']])
Y=np.ravel(Y) #change the shape of y to (n_samples, ) to suitable with LR model in Sklearn



######### Split data into training and test set #########
#split our dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split( X_list, Y, test_size=0.2, random_state=4)
#Sklearn train_test_split function ignores the original sequence of numbers. After a split, they can be presented in a different order

print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)



########## pipeline of plastic-type classifier ##############
pca=PCA(0.95) #no. of PCs explained 95% of variance
LRCV=LogisticRegressionCV(solver='lbfgs',penalty ='l2',multi_class='ovr',
                              max_iter=5000,random_state=4,n_jobs=2)
scaler = StandardScaler()
model_3 = Pipeline([("scaled",scaler),("PCA", pca),("LRCV",LRCV )])

###### cross_validation #########
cross_validation(model_3)
####### final training #######
def final_training_multiclass (model):
    start_time = time.time()

    model.fit(X_train,y_train)
    acc_train= model.score(X_train, y_train)
    acc_test= model.score(X_test, y_test)
    print ('Training accuracy:', round((acc_train)*100,2))
    print ('Test accuracy:', round((acc_test)*100,2))
    recal_train=recall_score(y_train,model.predict(X_train),average='macro')
    recal_test=recall_score(y_test,model.predict(X_test),average='macro')
    print ('Training recall:', round((recal_train),2))
    print ('Test recall:', round((recal_test),2))
    pre_train=precision_score(y_train,model.predict(X_train),average='macro')
    pre_test=precision_score(y_test,model.predict(X_test),average='macro')
    print ('Training pre:', round((pre_train),2))
    print ('Test pre:', round((pre_test),2))
    t=(time.time() - start_time)/60
    print ("My program took", t , "to run (mins)")
    
final_training_multiclass (model_3)
# save the model to disk
filename = 'E:/destop/1st measurement PL/model_3.sav'
# Save the model as a pickle in a file 
joblib.dump(model_3, filename)

#Check how many PCs explained 95% of variance 
print('no. of PCs explained 95% of variance', pca.components_.shape[0])


################### Compute confusion matrix of training #############################
loaded_model= joblib.load(filename)
yhat=loaded_model.predict(X_train) 
labels=np.ravel(np.array(df1[['code']] ))  
cnf_matrix = confusion_matrix(y_train, yhat, labels=labels)
np.set_printoptions(precision=2)

plt.figure(figsize=(10,8))
plot_confusion_matrix(cnf_matrix, classes=sample_labels,normalize= False,  title='Confusion matrix')


######################ERROR ANALYSIS#########

taxis=np.load('E:/spectra/sample/taxis.npy') #x_axis   

X=X_train
y=y_train
yhat=loaded_model.predict(X)
find_index=np.where(yhat != y) #find index of wrong prediction
find_index=np.ravel(find_index)

print('the number of wrong prediction: ',len(find_index))

true_label=[]
prec_label=[]


for i in find_index:
    for j in range(len(sample_labels)):
        if yhat[i]  == df1['code'][j]:
            prec_label.append(df1['sample_name'][j])
    norm=X[i]
    index_in_x_list=int(np.ravel(np.where(np.all(norm==X_list,axis=1))))
    name=df_y1.iloc[index_in_x_list]['sample_name']
    true_label.append(name)

wrong_pre_sample=np.unique(true_label)
spectra_lists = dict() #create a dictionary for index of wrong prediction 
                        #with the key is the true label and the value is the index to plot the spectra
pre_spectra_list= dict() #create a dictionary for reference strectra
                        #with the key is the true label and the value is the predicted label 
for i in range(len(wrong_pre_sample)):
    #print(wrong_pre_sample[i])
    spectra_lists[wrong_pre_sample[i]]=[]
    pre_spectra_list[wrong_pre_sample[i]]=[]
    for j in range(len(true_label)):
        if true_label[j]==wrong_pre_sample[i]:
           spectra_lists[wrong_pre_sample[i]].append(find_index[j]) 
           pre_spectra_list[wrong_pre_sample[i]].append(prec_label[j])
           
ref_sample_list= dict() #create a dictionary for unique value in reference strectra
                        #with the key is the true label and the value is the unique predicted label 
for i in pre_spectra_list.keys():
    ref_sample_list[i]=[]             
    ref_sample_list[i]=unique(pre_spectra_list[i])
    

#plot spectra based on spectra_lists and ref_sample_list
path='E:/destop/1st measurement PL/Spectra/norm/'
for i in spectra_lists.keys():
    fig,ax = plt.subplots(figsize=(10,8))
    for j in range(len(spectra_lists[i])):
        index_spec=spectra_lists[i][j]         
        ax.plot(taxis[658:2681],X[index_spec],label=i,color='black',linewidth=1)
    #load reference spectra
  
    for k in range (len(ref_sample_list[i])):
        name_ref_spec=ref_sample_list[i][k]
    
            
        for v in ['BAM','Brett','BPI','VG']: #4 is the number of company
            if name_ref_spec=='PE' and v=='Brett':
                name_ref_spec='LDPE'
                
            elif name_ref_spec=='PE' and v=='VG':
                name_ref_spec='PE1000'
            else:
                name_ref_spec=ref_sample_list[i][k]
            ref_predicted1= np.load(path+name_ref_spec+'_'+v+'.npy')                
            mean_pl1=np.mean(ref_predicted1,0)
            std_pl1 = np.std(ref_predicted1,0)
            ax.plot(taxis[658:2681], mean_pl1,linewidth=1,label=name_ref_spec+'_'+v)   
            ax.fill_between(taxis[658:2681], mean_pl1-std_pl1, mean_pl1+std_pl1, alpha=0.5)
    
        # Put a legend below current axis
    legend_without_duplicate_labels(ax)
    
    #ax.legend()
    plt.grid(b=True, which='major', color='k', linestyle='-', alpha=0.2)
    plt.grid(b=True, which='minor', color='b', linestyle='--', alpha=0.1)
    plt.minorticks_on()
    plt.xlabel('Wavelength (nm)',fontsize=15)
    plt.xticks(fontsize=13)
    plt.ylabel('Intensity (arb.u.)',fontsize=15)
    plt.yticks(fontsize=13)





############ PREDICTION ON MIXED SAMPLE ###############
img_file=  cbook.get_sample_data('E:/destop/1st measurement PL/c.jpg')
img=plt.imread(img_file)
mixed_data_pl=np.load('E:/destop/1st measurement PL/mix_sample_pl.npy') #loading plastic spectra
mixed_data_all=np.load('E:/destop/1st measurement PL/mixed_spectra.npy') #loading spectra)
x=np.load('E:/destop/1st measurement PL/x.npy') #x coordinates
y=np.load('E:/destop/1st measurement PL/y.npy') #y coordinates
x_pl=[]
y_pl=[]
for i in range(len(mixed_data_all)):
    for j in range (len(mixed_data_pl)):
        if np.all (mixed_data_all[i] == mixed_data_pl[j]):
            
            x_pl.append(x[i])
            y_pl.append(y[i])
    

        
y_true_plastic= np.load('E:/destop/1st measurement PL/y_true_pl.npy') #loeading true label

######## 
filename = 'E:/destop/1st measurement PL/model_3.sav'
loaded_model=joblib.load(filename)
yhat = loaded_model.predict(mixed_data_pl)
find_index_true=np.where(yhat != y_true_plastic) #find index of wrong prediction
find_index_true=np.ravel(find_index_true)
print('the number of wrong prediction: ',len(find_index_true))
print('accuracy', (len(y_true_plastic)-len(find_index_true))/len(y_true_plastic))
###########
    
sample_name=['PVC','PMMA','PS','PP','PET','PE','PC','PA']
#convert categories to number
df=pd.DataFrame(sample_name,columns={'sample_name'}) #change the type of the column
#convert categories to numbers
df.sample_name = pd.Categorical(df.sample_name)
df['code'] = df.sample_name.cat.codes
print(df) 
#############


##############plot result##############
colors = ['blue','black','red','pink','gray','purple','orange','turquoise'] # 8 colors = 8 plastic types
color_for_mapping=[]
label_for_mapping=[]

for i in range (len(yhat)):
    for j in range(len(sample_name)):
        if yhat[i] == df['code'][j]:
            label=df['sample_name'][j]
            color=colors[j]            
    color_for_mapping.append(color)
    label_for_mapping.append(label)


# Create a figure. Equal aspect so circles look circular
fig,ax = plt.subplots(figsize=(15,10))
ax.set_aspect('equal')
ax.set_axis_off()
fig.add_axes(ax)
# Show the image
ax.imshow(img) 

# Now, loop through coord arrays, and create a circle at each x,y pair
for xx,yy,aa,bb in zip(x_pl,y_pl,color_for_mapping,label_for_mapping):
    circ = Circle((xx,yy),20,color=aa,label=bb)
    ax.add_patch(circ)
# Put a legend below current axis
legend_without_duplicate_labels(ax)
# Show the image
plt.show()
    
  
    
