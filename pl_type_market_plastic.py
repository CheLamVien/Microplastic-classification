# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 20:00:36 2021

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
    ax.legend(*zip(*unique),loc='upper center',bbox_to_anchor=(0.5, 1.1),
         fancybox=True, shadow=True, ncol=4,fontsize=12)
################3
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



sample_name=plastic_name_1+plastic_name_2+plastic_name_3+plastic_name_4+plastic_name_5+plastic_name_6+plastic_name_7+plastic_name_8+plastic_name_9+plastic_name_10+plastic_name_11+plastic_name_12+plastic_name_13


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
                   'PS','PS','PS','PS',
                   'PE','PE','PE','PE','PE','PE','PE',
                   'PE','PE','PE','PE','PE',
                   'PET','PET','PET','PET','PET','PET','PET','PET','PET','PET','PET','PET','PET','PET',
                   'PP','PP','PP','PP','PP','PP','PP','PP','PP','PP','PP','PP','PP','PP','PP',
                   ]

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

################### First pipeline ########
pca=PCA(0.95) #no. of PCs explained 95% of variance
'''
LRCV=LogisticRegressionCV(solver='lbfgs',penalty ='l2',multi_class='ovr',
                              max_iter=5000,random_state=4,n_jobs=2)
'''
LRCV=LogisticRegressionCV(solver='newton-cg',penalty ='l2',multi_class='multinomial',
                              max_iter=5000,random_state=4,n_jobs=2)
scaler = StandardScaler()
model_6 = Pipeline([("scaled",scaler),("PCA", pca),("LRCV",LRCV )])



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

cross_validation(model_6)

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
    
final_training_multiclass (model_6)
# save the model to disk
filename = 'E:/destop/1st measurement PL/model_8.sav'
# Save the model as a pickle in a file 
joblib.dump(model_6, filename)

#Check how many PCs explained 95% of variance 
print('no. of PCs explained 95% of variance', pca.components_.shape[0])


################### Compute confusion matrix #############################3
loaded_model= joblib.load(filename)
#yhat=loaded_model.predict(X_test_pca)
yhat=loaded_model.predict(X_train)


############
labels=np.ravel(np.array(df1[['code']] ))  
#cnf_matrix = confusion_matrix(y_test, yhat, labels=labels)
cnf_matrix = confusion_matrix(y_train, yhat, labels=labels)
np.set_printoptions(precision=2)

plt.figure(figsize=(10,8))
plot_confusion_matrix(cnf_matrix, classes=sample_labels,normalize= False,  title='Confusion matrix')



######################ERROR ANALYSIS##########################

taxis=np.load('E:/spectra/sample/taxis.npy') #x_axis   
loaded_model= joblib.load(filename)

X=X_train #test on test
y=y_train



yhat=loaded_model.predict(X)
find_index=np.where(yhat != y) #find index of wrong prediction 
find_index=np.ravel(find_index)
print('the number of wrong prediction: ',len(find_index))

true_label=[]
prec_label=[]


#np.where(np.all(X_list[916]==X_test_non_scaled,axis=1))
for i in find_index:
    for j in range(len(sample_labels)):
        if yhat[i]  == df1['code'][j]:
            prec_label.append(df1['sample_name'][j])
    norm=X[i]
    index_in_x_list=int(np.ravel(np.where(np.all(norm==X_list,axis=1)))[0]) #[0] because there are duplicate spectra (spectra are the same so just take the first one)
    #print(np.ravel(np.where(np.all(norm==X_list,axis=1))))                  # spectrum with index 916 is the same as 914
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
    
#spectra_lists['N1'][1] 
#vẽ biểu đồ dựa trên spectra_lists and ref_sample_list
path='E:/destop/1st measurement PL/Spectra/norm/'
for i in spectra_lists.keys():
    fig,ax = plt.subplots(figsize=(10,8))
    for j in range(len(spectra_lists[i])):
        index_spec=spectra_lists[i][j]         
        ax.plot(taxis[658:2681],X[index_spec],label=i,color='black',linewidth=2)
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
            ax.plot(taxis[658:2681], mean_pl1,linewidth=0.5,label=name_ref_spec+'_'+v)   
            ax.fill_between(taxis[658:2681], mean_pl1-std_pl1, mean_pl1+std_pl1, alpha=0.5)
    
        # Put a legend below current axis
    
    
    #ax.legend()
    legend_without_duplicate_labels(ax)
    plt.grid(b=True, which='major', color='k', linestyle='-', alpha=0.2)
    plt.grid(b=True, which='minor', color='b', linestyle='--', alpha=0.1)
    plt.minorticks_on()
    plt.xlabel('Wavelength (nm)',fontsize=rcParams['axes.labelsize'])
    plt.xticks(fontsize=rcParams['axes.labelsize'])
    plt.ylabel('Intensity (arb.u.)',fontsize=rcParams['axes.labelsize'])
    plt.yticks(fontsize=rcParams['axes.labelsize'])


############ PREDICTION ON MIXED SAMPLE ###############
img_file=  cbook.get_sample_data('E:/destop/1st measurement PL/c.jpg')
img=plt.imread(img_file)
mixed_data_pl=np.load('E:/destop/1st measurement PL/mix_sample_pl.npy') #loading spectra
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
filename = 'E:/destop/1st measurement PL/model_8.sav'
loaded_model=joblib.load(filename)
yhat = loaded_model.predict(mixed_data_pl)
find_index_true=np.where(yhat != y_true_plastic) #find index of wrong prediction
find_index_true=np.ravel(find_index_true)
print('the number of wrong prediction: ',len(find_index_true))
print('accuracy', (len(y_true_plastic)-len(find_index_true))/len(y_true_plastic))


################ GRID SEARCH ###########
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