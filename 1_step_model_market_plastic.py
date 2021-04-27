# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 20:03:40 2021

@author: Vien Che
"""
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
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
from sklearn.metrics import precision_score,recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
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
         fancybox=True, shadow=True, ncol=8,fontsize=12)
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



nonpl_name_1=['N1_nature','N2_nature','N3_nature','N4_nature']
nonpl_name_2=['N5inner_nature','N5outer_nature','N6inner_nature','N6outer_nature']
nonpl_name_3=['N7_nature','N8_nature','N9_nature','N10_nature']


    
    
sample_name=plastic_name_1+plastic_name_2+plastic_name_3+plastic_name_4+plastic_name_5+plastic_name_6+plastic_name_7+plastic_name_8+plastic_name_9+plastic_name_10+plastic_name_11+plastic_name_12+plastic_name_13+nonpl_name_1+nonpl_name_2+nonpl_name_3

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
                   'Nature sample','Nature sample','Nature sample','Nature sample',
                   'Nature sample','Nature sample','Nature sample','Nature sample',
                   'Nature sample','Nature sample','Nature sample','Nature sample'
                   ]

sample_labels=['PVC','PMMA','PS','PP','PET','PE','PC','PA','Nature sample']

#convert plastic types (categories) into number
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
#################


#############SMOOTHEN##########
#X_list= scipy.signal.savgol_filter(X_list, 11, 2) # window size 51, polynomial order 3




#########Target vector for plastic and non-plastic classification #########
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
df_y1['sample_type'] = pd.Categorical(df_y1.sample_type)
#convert categories to numbers
df_y1['code'] = df_y1.sample_type.cat.codes #code for plastic and non plastic classification


#change the shape of y to (n_samples, ) to suitable with LR model in Sklear
Y=np.ravel(np.array(df_y1[['code']]))

##########MEDIAN NORMALIZATION#############
'''
median_list=np.median(X_list, axis=0)
X_list_median_norm=X_list-median_list
'''



####################################split data #############

X_train, X_test, y_train, y_test = train_test_split( X_list, Y, test_size=0.2, random_state=4)
'''
#Sklearn train_test_split function ignores the original sequence of numbers. After a split, they can be presented in a different order
X_train_non_scaled, X_test_non_scaled, y_train, y_test = train_test_split(X_list_median_norm, Y, 
                                                            test_size=0.1, random_state=4)
'''


print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


################### pipeline ########
pca=PCA(0.95) #no. of PCs explained 95% of variance
LRCV=LogisticRegressionCV(solver='lbfgs',penalty ='l2',multi_class='ovr',
                              max_iter=5000,random_state=4,n_jobs=2)
scaler = StandardScaler()
model_7 = Pipeline([("scaled",scaler),("PCA", pca),("LRCV",LRCV )])



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

cross_validation(model_7)

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
    
final_training_multiclass (model_7)
# save the model to disk
filename = 'E:/destop/1st measurement PL/model_7.sav'
# Save the model as a pickle in a file 
joblib.dump(model_7, filename)

#Check how many PCs explained 95% of variance 
print('no. of PCs explained 95% of variance', pca.components_.shape[0])


################### Compute confusion matrix #############################3
loaded_model= joblib.load(filename)
#yhat=loaded_model.predict(X_test_pca)
yhat=loaded_model.predict(X_train)
y=y_train

############
labels=np.ravel(np.array(df[['code']] ))  
cnf_matrix = confusion_matrix(y, yhat, labels=labels)
np.set_printoptions(precision=2)

plt.figure(figsize=(10,8))
plot_confusion_matrix(cnf_matrix, classes=sample_labels,normalize= False,  title='Confusion matrix')


########calculate the plastic vs non-plastic accuracy ##########
y_true_pl_vs_non_pl=[]
y=y_test
yhat=loaded_model.predict(X_test)
for i in y:
    if i == 0:
        i=0
    else:
        i=1
    y_true_pl_vs_non_pl.append(i)

y_hat_pl_vs_non_pl=[]
for i in yhat:
    if i == 0:
        i=0
    else:
        i=1
    y_hat_pl_vs_non_pl.append(i)

index_true=np.where(np.array(y_hat_pl_vs_non_pl) != np.array(y_true_pl_vs_non_pl)) #find index of wrong prediction
index_true=np.ravel(index_true)
print('the number of wrong prediction: ',len(index_true))
print('accuracy', (len(y_true_pl_vs_non_pl)-len(index_true))/len(y_true_pl_vs_non_pl))
###########calculate the plastic-type accuracy #########3
y_plastic_true=[]
in_pl=[]
for i in range(len(y)):
    if y[i] != 0:
        y_plastic_true.append(y[i])
        in_pl.append(i)
        
y_plastic_hat=[]
for i in in_pl:
    y_plastic_hat.append(yhat[i])
        
index=np.where(np.array(y_plastic_hat) != np.array(y_plastic_true)) #find index of wrong prediction
index=np.ravel(index)
print('the number of wrong prediction: ',len(index))
print('accuracy', (len(y_plastic_true)-len(index))/len(y_plastic_true))       


############ PREDICTION ON MIXED SAMPLE ###############
img_file=  cbook.get_sample_data('E:/destop/1st measurement PL/c.jpg')
img=plt.imread(img_file)
mixed_data_all=np.load('E:/destop/1st measurement PL/mixed_spectra.npy') #loading spectra)
x=np.load('E:/destop/1st measurement PL/x.npy') #x coordinates
y=np.load('E:/destop/1st measurement PL/y.npy') #y coordinates        
y_true= np.load('E:/destop/1st measurement PL/y_true_1_step.npy') #loading true label

######## 
filename = 'E:/destop/1st measurement PL/model_9.sav'
loaded_model=joblib.load(filename)
yhat = loaded_model.predict(mixed_data_all)
find_index_true=np.where(yhat != y_true) #find index of wrong prediction
find_index_true=np.ravel(find_index_true)
print('the number of wrong prediction: ',len(find_index_true))
print('accuracy', (len(y_true)-len(find_index_true))/len(y_true))

########calculate the plastic vs non-plastic accuracy ##########
y_true_pl_vs_non_pl=[]
for i in y_true:
    if i == 0:
        i=0
    else:
        i=1
    y_true_pl_vs_non_pl.append(i)

y_hat_pl_vs_non_pl=[]
for i in yhat:
    if i == 0:
        i=0
    else:
        i=1
    y_hat_pl_vs_non_pl.append(i)

index_true=np.where(np.array(y_hat_pl_vs_non_pl) != np.array(y_true_pl_vs_non_pl)) #find index of wrong prediction
index_true=np.ravel(index_true)
print('the number of wrong prediction: ',len(index_true))
print('accuracy', (len(y_true_pl_vs_non_pl)-len(index_true))/len(y_true_pl_vs_non_pl))
###########calculate the plastic-type accuracy #########3
y_plastic_true=[]
in_pl=[]
for i in range(len(y_true)):
    if y_true[i] != 0:
        y_plastic_true.append(y_true[i])
        in_pl.append(i)
        
y_plastic_hat=[]
for i in in_pl:
    y_plastic_hat.append(yhat[i])
        
index=np.where(np.array(y_plastic_hat) != np.array(y_plastic_true)) #find index of wrong prediction
index=np.ravel(index)
print('the number of wrong prediction: ',len(index))
print('accuracy', (len(y_plastic_true)-len(index))/len(y_plastic_true))       

########## PLOTTING ###########
sample_name=['PVC','PMMA','PS','PP','PET','PE','PC','PA','Natural material']

#convert plastic types (categories) into number
df=pd.DataFrame(sample_name,columns={'sample_name'}) #change the type of the column
df.sample_name = pd.Categorical(df.sample_name)
df['code'] = df.sample_name.cat.codes
print(df)
#############


##############plot result##############
colors = ['blue','black','red','pink','gray','purple','orange','turquoise','blue'] 
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
for xx,yy,aa,bb in zip(x,y,color_for_mapping,label_for_mapping):
    if 'Natural material' in bb:
        ax.scatter(xx,yy,color=aa,label=bb,marker='*',s=80)
    else:
        ax.scatter(xx,yy,color=aa,label=bb,marker='o',s=80)

# Put a legend below current axis
legend_without_duplicate_labels(ax)
# Show the image
plt.show()






################### pipeline FOR NN ########
pca=PCA(0.95) #no. of PCs explained 95% of variance
NN= MLPClassifier(hidden_layer_sizes=(13,13),max_iter=5000,random_state=1,alpha=1.5) #train with NN random_state to preproduce the same result
#can set verbose=1 to see the value of loss function 
scaler = StandardScaler()
model_9 = Pipeline([("scaled",scaler),("PCA", pca),("NN",NN )])


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
    
    
    
cross_validation(model_9)

final_training_multiclass (model_9)
# save the model to disk
filename = 'E:/destop/1st measurement PL/model_9.sav'
# Save the model as a pickle in a file 
joblib.dump(model_9, filename)

#Check how many PCs explained 95% of variance 
print('no. of PCs explained 95% of variance', pca.components_.shape[0])



######################ERROR ANALYSIS##########################

taxis=np.load('E:/spectra/sample/taxis.npy') #x_axis   
loaded_model= joblib.load(filename)

#plot the wrong prediction spectrum

X=X_test #test on train
y=y_test


yhat=loaded_model.predict(X)
find_index=np.where(yhat != y) #find index of wrong prediction 
find_index=np.ravel(find_index)
print('the number of wrong prediction: ',len(find_index))

true_label=[]
prec_label=[]


#np.where(np.all(X_list[916]==X_test_non_scaled,axis=1))
for i in find_index:
    for j in range(len(sample_labels)):
        if yhat[i]  == df['code'][j]:
            prec_label.append(df['sample_name'][j])
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
#plot spectra based on spectra_lists and ref_sample_list
path='E:/destop/1st measurement PL/Spectra/norm/'
for i in spectra_lists.keys():
    fig,ax = plt.subplots(figsize=(10,8))
    for j in range(len(spectra_lists[i])):
        index_spec=spectra_lists[i][j]         
        ax.plot(taxis[658:2681],X[index_spec],label=i,color='black',linewidth=2)
    #load reference spectra
  
    for k in range (len(ref_sample_list[i])):
        name_ref_spec=ref_sample_list[i][k]
        if name_ref_spec == 'PVC':
            compa_list=['BAM','BPI','VG'] #4 is the number of company
        elif name_ref_spec == 'Nature sample':
            break
        else: 
            compa_list=['BAM','Brett','BPI','VG']
        for v in compa_list: 
        
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
    legend_without_duplicate_labels(ax)
    
    #ax.legend()
    plt.grid(b=True, which='major', color='k', linestyle='-', alpha=0.2)
    plt.grid(b=True, which='minor', color='b', linestyle='--', alpha=0.1)
    plt.minorticks_on()
    plt.xlabel('Wavelength (nm)',fontsize=rcParams['axes.labelsize'])
    plt.xticks(fontsize=rcParams['axes.labelsize'])
    plt.ylabel('Intensity (arb.u.)',fontsize=rcParams['axes.labelsize'])
    plt.yticks(fontsize=rcParams['axes.labelsize'])




################### Compute confusion matrix #############################3
loaded_model= joblib.load(filename)
#yhat=loaded_model.predict(X_test_pca)
X=X_test
y=y_test

yhat=loaded_model.predict(X)


############
labels=np.ravel(np.array(df[['code']] ))  
#cnf_matrix = confusion_matrix(y_test, yhat, labels=labels)
cnf_matrix = confusion_matrix(y, yhat, labels=labels)
np.set_printoptions(precision=2)

plt.figure(figsize=(10,8))
plot_confusion_matrix(cnf_matrix, classes=sample_labels,normalize= False,  title='Confusion matrix')









#####################################TRAIN WITH DIFFERENT NUMBER OF PC##############
####loading mixed sample data##########
mixed_data_all=np.load('E:/destop/1st measurement PL/mixed_spectra.npy') #loading spectra)      
y_true= np.load('E:/destop/1st measurement PL/y_true_1_step.npy') #loading true label


acc_train_list=[]
acc_test_list=[]
acc_mix=[]

for i in range (1,40):
    pca=PCA(n_components=i,random_state=1) #no. of PCs explained 95% of variance
    NN= MLPClassifier(hidden_layer_sizes=(13,13),max_iter=5000,random_state=1,alpha=1.5) #train with NN random_state to preproduce the same result
    #can set verbose=1 to see the value of loss function 
    scaler = StandardScaler()
    model = Pipeline([("scaled",scaler),("PCA", pca),("NN",NN )])
    model.fit(X_train,y_train)
    acc_train= model.score(X_train, y_train)
    acc_test= model.score(X_test, y_test)
    acc_train_list.append(acc_train)
    acc_test_list.append(acc_test)
    yhat=model.predict(mixed_data_all)
    find_index_true=np.where(yhat != y_true) #find index of wrong prediction
    find_index_true=np.ravel(find_index_true)
    acc_mix.append((len(y_true)-len(find_index_true))/len(y_true))
    print ('Training accuracy:', acc_train)
    print ('Test accuracy:', acc_test)

c=['black','blue','red','black']
# Set the font properties (for use in legend)   
font_path = 'C:\Windows\Fonts\Arial.ttf'
array=np.arange(1,40,1)
no_component=[i for i in array]
plt.figure(figsize=(10,8))
plt.plot(no_component, acc_train_list,'--o',linewidth=2,color = 'red',label='Traning set')   
plt.plot(no_component, acc_mix,'--v',linewidth=2,color = 'black',label='Mixed sample') 
plt.plot(no_component, acc_test_list,'--*',linewidth=2,color = 'blue',label='Test set') 
plt.grid(b=True, which='major', color='k', linestyle='-', alpha=0.2)
plt.grid(b=True, which='minor', color='b', linestyle='--', alpha=0.1)
plt.minorticks_on()
plt.xlabel('The first d number of PCs',fontsize=15)
plt.xticks(fontsize=13)
plt.ylabel('Accuracy',fontsize=15)
plt.yticks(fontsize=13)
leg = plt.legend(loc='best', numpoints=1,fontsize=15)
for legobj in leg.legendHandles:
    legobj.set_linewidth(2.0)

##################################DETERMINE amount of variance that explained by 17 PCs #########
    
    
pca=PCA(n_components=17,random_state=1)   
scaler = StandardScaler()
pca_scale = Pipeline([("scaled",scaler),("PCA", pca)])
pca_scale.fit(X_train)
print('17 PCs explained',sum(pca.explained_variance_ratio_),' of variance' )

    
############################## train and save the NN model with 17 PC ########
pca=PCA(n_components=17,random_state=1) #no. of PCs explained 95% of variance
NN= MLPClassifier(hidden_layer_sizes=(13,13),max_iter=5000,random_state=1,alpha=1.5) #train with NN random_state to preproduce the same result
#can set verbose=1 to see the value of loss function 
scaler = StandardScaler()
model_10 = Pipeline([("scaled",scaler),("PCA", pca),("NN",NN )])

cross_validation(model_10)

final_training_multiclass (model_10)
# save the model to disk
filename = 'E:/destop/1st measurement PL/model_10.sav'
# Save the model as a pickle in a file 
joblib.dump(model_10, filename)






































