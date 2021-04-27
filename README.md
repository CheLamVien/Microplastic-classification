# Microplastic-classification
Logistic Regression and Neural network models for identification of microplastic based on their spectra.
Two different approaches can address a microplastic classification problem after reducing the number of features by PCA. 
The first method, which is named as 2-steps model, is to combine one binary classifier and one multiclass classifier. 
Two models are trained separately and then combined to create a full solution for the classification problem. 
First, the binary model, plastic vs. nonplastic classifier, is built to determine whether the unknown specimens are plastic or non-plastic. 
Then, the multiclass classifier with k=8 corresponding to 8 different plastic types is used to identify which specific type of plastic the samples are if they are plastic. 
This multiclass model is called a plastic-type classifier. 
The second approach is to solve the microplastic classification task with one multiclass classifier, which directly gives information about analyzed samples based on their spectra. 
The model in the second approach is called as 1-step model
.
