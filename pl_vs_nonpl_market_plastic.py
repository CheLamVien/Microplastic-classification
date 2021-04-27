# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:58:48 2021

@author: Vien Che
"""

import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
import itertools
import pandas as pd
import time
import joblib
from sklearn.model_selection import GridSearchCV
from matplotlib import rcParams
from sklearn.decomposition import PCA 
from sklearn.metrics import precision_score,recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

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
    ax.legend(*zip(*unique),loc='upper center',bbox_to_anchor=(0.5, -0.12),
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

plastic_name_10 =['HDPE_black_44','HDPE_blue_96','HDPE_grey_46'
                 ,'HDPE_pink_125','HDPE_white_10','HDPE_white_13','HDPE_yellow_52']
plastic_name_11 =['LDPE_blue_168','LDPE_white_6','LDPE_white_47','LDPE_yellow_169','LDPE_green_23']
plastic_name_12=['PET_black_42','PET_black_71','PET_blue_81','PET_blue_100',
'PET_pink_126','PET_purple_153','PET_white_160','PET_yellow_35','PET_yellow_156',
'PET_green_104','PET_green_113','PET_red_84','PET_transparent_78','PET_transparent_106']
plastic_name_13=['PP_black_26','PP_black_43','PP_black_131','PP_blue_37',
'PP_blue_97','PP_white_2','PP_white_14','PP_white_101','PP_yellow_19',
'PP_yellow_124','PP_green_75','PP_green_86','PP_red_163','PP_red_164','PP_transparent_140']

nonpl_name_1=['N1_nature','N2_nature','N3_nature','N4_nature']
nonpl_name_2=['N5inner_nature','N5outer_nature','N6inner_nature','N6outer_nature']
nonpl_name_3=['N7_nature','N8_nature','N9_nature','N10_nature']

    
sample_name=plastic_name_1+plastic_name_2+plastic_name_3+plastic_name_4+plastic_name_5+plastic_name_6+plastic_name_7+plastic_name_8+plastic_name_9+plastic_name_10+plastic_name_11+plastic_name_12+plastic_name_13+nonpl_name_1+nonpl_name_2+nonpl_name_3



#convert plastic types (categories) into number
sample_labels1=['Plastic','Non Plastic']
df1=pd.DataFrame(sample_labels1,columns={'sample_name'}) #change the type of the column
#convert categories to numbers
df1.sample_name = pd.Categorical(df1.sample_name)
df1['code'] = df1.sample_name.cat.codes
print(df1)



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
#################


#############SMOOTHEN##########
#X_list= scipy.signal.savgol_filter(X_list, 11, 2) # window size 51, polynomial order 3




#########Target vector for plastic and non-plastic classification #########

sample_name_taget_plastics=['Plastic' for i in range(0,36+41)] #36 types of virgin plastic from 4 companies and 41 types of colored plastics
sample_name_taget_non_plastics=['Non_Plastic' for i in range(0,12)] #12 types of non plastic
sample_name_taget1=sample_name_taget_plastics+sample_name_taget_non_plastics



Y_list1= [] #empty list for defining plastic or non-plastic
Y_list2= [] #empty list for defining sample names with the company

for i in range(len(sample_name_taget1)):
    name=[sample_name_taget1[i] for k in range(0,len(dataf[i]))]
    name1=[sample_name[i] for k in range(0,len(dataf[i]))]

    Y_list1.append(name)
    Y_list2.append(name1)


flattend_Y_list1= [item for sublist in Y_list1 for item in sublist]
flattend_Y_list2= [item for sublist in Y_list2 for item in sublist]

df_y1=pd.DataFrame({'sample_type':flattend_Y_list1,'sample_name':flattend_Y_list2}) 
df_y1['sample_type'] = pd.Categorical(df_y1.sample_type)#change the type of the column
#convert categories to numbers
df_y1['code'] = df_y1.sample_type.cat.codes #code for plastic and non plastic classification

Y=np.array(df_y1[['code']])
Y=np.ravel(Y)

####################################split data #############
X_train, X_test, y_train, y_test= train_test_split( X_list, Y, 
                                                            test_size=0.2, random_state=4)
#Sklearn train_test_split function ignores the original sequence of numbers. After a split, they can be presented in a different order


print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


################### First pipeline ########
pca=PCA(0.95) #no. of PCs explained 95% of variance
'''
#old
LRCV=LogisticRegressionCV(solver='lbfgs',penalty ='l2',multi_class='ovr',
                              max_iter=5000,random_state=4,n_jobs=2)
'''
LRCV=LogisticRegressionCV(solver='newton-cg',penalty ='l2',multi_class='ovr',
                              max_iter=10000,random_state=4,n_jobs=2)
scaler = StandardScaler()
model_5 = Pipeline([("scaled",scaler),("PCA", pca),("LRCV",LRCV )])



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

cross_validation(model_5)
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

final_training (model_5)
# save the model to disk
filename = 'E:/destop/1st measurement PL/model_5.sav'
# Save the model as a pickle in a file 
joblib.dump(model_5, filename)

#Check how many PCs explained 95% of variance 
print('no. of PCs explained 95% of variance', pca.components_.shape[0])

########## make prediction on mixed sample #########
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
plt.figure(figsize=(10,8))    
#plot the wrong prediction spectrum

#test_check
X=X_train
y=y_train


yhat=loaded_model.predict(X)
find_index=np.where(yhat != y) #find index of wrong prediction
find_index=np.ravel(find_index)
print('the number of wrong prediction: ',len(find_index))


true_label=[]
prec_label=[]
name_wrong=[]
for i in find_index:
    true_label.append(y[i])
    prec_label.append(yhat[i])
    for j in range(len(sample_labels1)):
        if y[i] == df1['code'][j]:
            label_true=df1['sample_name'][j]
    
    norm=X[i]
    index_in_x_list=int(np.ravel(np.where(np.all(norm==X_list,axis=1))))
    name=df_y1.iloc[index_in_x_list]['sample_name']
    name_wrong.append(name)
    #norm=X_test_non_scaled[i]/np.max(X_test_non_scaled[i])
    plt.plot(taxis[658:2681],norm,label=name,linewidth=2) #plot all spectra of each sample with non nomaliyation
    plt.legend() 
    plt.grid(b=True, which='major', color='k', linestyle='-', alpha=0.2)
    plt.grid(b=True, which='minor', color='b', linestyle='--', alpha=0.1)
    plt.minorticks_on()
    plt.xlabel('Wavelength (nm)',fontsize=rcParams['axes.labelsize'])
    plt.xticks(fontsize=rcParams['axes.labelsize'])
    plt.ylabel('Intensity (arb.u.)',fontsize=rcParams['axes.labelsize'])
    plt.yticks(fontsize=rcParams['axes.labelsize'])

    leg = plt.legend(loc=1, numpoints=1, ncol=5)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
        

def countX(lst, x): 
    return lst.count(x)
for i in unique(name_wrong):
    print('{} has occurred {} times'.format(i, countX(name_wrong, i))) 



####################### GRID SEARCH ##################

pca=PCA(0.95) #no. of PCs explained 95% of variance
LRCV=LogisticRegressionCV(max_iter=100000,random_state=4,tol=0.01,cv=3)
scaler = StandardScaler()
pipe = Pipeline(steps=[("scaled",scaler),("PCA", pca),("LRCV",LRCV )])

param_grid = {
    'LRCV__penalty': ('l2', 'l1','elasticnet'),
    'LRCV__solver': ('newton-cg', 'lbfgs','liblinear','sag','saga'),
    'LRCV__multi_class': ('ovr', 'multinomial'),
    'LRCV__l1_ratios':[None,[0.1,0.3,0.5,0.7,0.9]]}

search = GridSearchCV(pipe, param_grid)
search.fit(X_train, y_train)
print("Best parameter (CV score=%0.4f):" % search.best_score_)
print(search.best_params_)
cvres = search .cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)
#'newton-cg' and  'lbfgs' has the same validation accuracy