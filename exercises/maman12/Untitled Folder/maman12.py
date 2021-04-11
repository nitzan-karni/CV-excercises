#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
import numpy as np
import matplotlib.pylab as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[155]:


labels_path = 'train-labels-idx1-ubyte.gz'
images_path = 'train-images-idx3-ubyte.gz'

# Read the images and labels
with gzip.open(labels_path, 'rb') as lpath:
    labels = np.frombuffer(lpath.read(), dtype=np.uint8, offset=8)
with gzip.open(images_path, 'rb') as ipath:
    images = np.frombuffer(ipath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)


# In[156]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))


# In[157]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[158]:


# show 16 smaples from the train set
plt.figure(figsize=(16,8))
n = 12
for i in range(0,n):
    plt.subplot(3,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    im = images[i].reshape([28,28])
    label = class_names[labels[i]]
    plt.imshow(im, cmap='gray', interpolation='nearest')
    plt.title(label)
plt.show()


# # Part 1 Q1

# In[159]:


# Split to train set and validation set
xt, xv, yt, yv = train_test_split(images, labels, random_state=0, train_size=0.2, test_size=0.06)
scores = []

# For every n_neighbors param train the model and get the score 
for n in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(xt,yt)
    scores.append(knn.score(xv, yv))
fig = plt.figure()
plt.plot(list(range(1,11)), scores)
plt.xlabel("n_neighbors")
plt.ylabel("score")
plt.title("KNN for differnet N values")
fig.savefig("KNN for differnet N values.png")


# In[185]:


print("Maximum score is {}, maximized by K={}".format(max(scores), np.argmax(np.array(scores))+1))


# # Part 1 Q2

# b)

# In[197]:


pca = PCA(n_components=256)
imagesProjected = pca.fit_transform(images)


# In[198]:


mean = pca.mean_
fig = plt.figure()
plt.title("Mean image")
plt.imshow(mean.reshape([28,28]), cmap='gray', interpolation='nearest')
fig.savefig("Q2b-Mean image.png")


# In[199]:



fig = plt.figure(figsize=(16,8))
n = 6
for i in range(0,n):
    plt.subplot(2,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    im = pca.components_[i].reshape([28, 28])
    label = class_names[labels[i]]
    plt.imshow(im)
    plt.title("Principal Component {}".format(i+1))
plt.show()

fig.savefig("Q2C - Principal components.png")


# c)

# In[200]:


fig = plt.figure()
plt.plot(pca.explained_variance_ratio_)
plt.title("Explained total variance")
fig.savefig("Q2C-Explained total variance.png")


# d)

# In[201]:


sum95 = 0
sum80 = 0
notfound80 = True
i = 0

# Calculate the number of bases needed to reach % of varuance
while sum95 < 0.95:
    sum95 += pca.explained_variance_ratio_[i]
    if sum80 >= 0.8 and notfound80:
        print("Number of bases needed to reach 80% variance: {}".format(i+1))
        notfound80 = False
    else:
        if notfound80:
            sum80 += pca.explained_variance_ratio_[i]
    i +=1
print("Number of bases needed to reach 95% variance: {}".format(i+1))


# e)

# In[170]:


pca = PCA(n_components=2)
imagesPCA2 = pca.fit_transform(images)


# In[171]:


fig = plt.figure(figsize=(16,8))
plt.scatter(imagesPCA2[:,0], imagesPCA2[:,1], c=labels)
plt.colorbar()
plt.title("PCA 2D projection")
fig.savefig("Q2E-PCA 2D projection.png")


# f)

# In[172]:


fig = plt.figure(figsize=(12,8))
i = 1
for N in [2, 10, 20]:
    plt.subplot(1,3,i, sharey=fig.gca())
    pca = PCA(n_components=N)
    
    # Project data to ND dimension
    imagesPCA = pca.fit_transform(images)
    xt, xv, yt, yv = train_test_split(imagesPCA, labels, random_state=0, train_size=0.2, test_size=0.06)
    scores = []
    for n in range(1,11):
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(xt,yt)
        scores.append(knn.score(xv, yv))
    plt.plot(list(range(1,11)), scores)
    plt.title("KNN projected PCA by {}".format(N))
    i += 1
fig.savefig("Q2F-KNNPCA.png")


# g)

# In[173]:


img = images[0]
im = img.reshape([28,28])
fig = plt.figure(figsize=(9, 9))
plt.subplot(3,3,1)
plt.xticks([])
plt.yticks([])
plt.grid(False)
label = class_names[labels[0]]
plt.imshow(im)
plt.title("{} - original".format(label))

i = 2
for k in [150, 100,50,10,5,2]:
    pca = PCA(n_components=k)
    pca.fit(images)
    # Project image to k dimension and recreate it from the projection
    img_pca = pca.inverse_transform(pca.transform([img]))
    im = img_pca.reshape([28,28])
    plt.subplot(3,3,i)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(im)
    plt.title("K={}".format(k))
    i += 1
    
fig.savefig("Q2G-PCAREVERSE.png")


# # Part 1 Q3 - LDA

# b)

# In[175]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=9)
lda.fit(images, labels)


# c)

# In[176]:


# c
fig = plt.figure()
plt.plot(lda.explained_variance_ratio_)
plt.title("Explained variance ratio")
fig.savefig("Q3C.png")


# In[177]:


# d
sum95 = 0
sum80 = 0
notfound80 = True
i = 0
while sum95 < 0.95:
    sum95 += lda.explained_variance_ratio_[i]
    if sum80 >= 0.8 and notfound80:
        print("Number of bases needed to reach 80% variance: {}".format(i+1))
        notfound80 = False
    else:
        if notfound80:
            sum80 += lda.explained_variance_ratio_[i]
    i +=1
print("Number of bases needed to reach 95% variance: {}".format(i+1))


# In[180]:


# e
lda = LinearDiscriminantAnalysis(n_components=2)
imagesLDA2 = pca.fit_transform(images, labels)

fig = plt.figure(figsize=(16,8))
plt.scatter(imagesLDA2[:,0], imagesLDA2[:,1], c=labels)
plt.colorbar()
plt.title("LDA projected 2D")
fig.savefig("Q3E.png")


# In[179]:


# f

fig = plt.figure(figsize=(12,8))
i = 1
for N in [2, 9]:
    plt.subplot(1,3,i, sharey=fig.gca())
    lda = LinearDiscriminantAnalysis(n_components=N)
    imagesLDA = lda.fit_transform(images, labels)
    xt, xv, yt, yv = train_test_split(imagesLDA, labels, random_state=0, train_size=0.2, test_size=0.06)
    scores = []
    for n in range(1,11):
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(xt,yt)
        scores.append(knn.score(xv, yv))
    plt.plot(list(range(1,11)), scores)
    plt.title("KNN projected LDA by {}".format(N))
    i += 1
fig.savefig("Q3F.png")


# # Part 2
# # Q1

# In[2]:


import cv2


# In[3]:


# section 1 train denseSIFT
sift = cv2.SIFT_create()


# In[4]:


import os

# Read images from folder and return labels and images ndarray
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            labels.append(filename.split('_')[0])
    return (images, labels)


# In[5]:


images, labels = load_images_from_folder('spatial_envelope_256x256_static_8outdoorcategories')


# In[6]:


step_size = int(len(images[0]) ** (3/8))

# Matrix of images x descriptors
desc = np.ndarray((len(images),1024,128))
i = 0
for image in images:
    # Create grid for dense SIFT
    kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, image.shape[0], step_size) 
                                    for x in range(0, image.shape[1], step_size)]
    # Compute denseSIFT descriptors for the grid
    (pts, descs) = sift.compute(image, kp)
    desc[i] = descs
    i += 1


# In[7]:


# Spread to array of total descriptors
X = desc.reshape(desc.shape[0], -1)


# In[8]:


label2Num = {'coast': 1, 'forest': 2, 'highway': 3, 'insidecity': 4, 'mountain': 5, 'opencountry': 6, 'street': 7, 'tallbuilding': 8}


# In[9]:


labelNums = []

# Create numberic values for labels
for label in labels:
    labelNums.append(label2Num[label])

labelNums = np.array(labelNums)


# In[10]:


# reshape descriptiors data to contain only descriptors
X = desc.reshape(desc.shape[0]*desc.shape[1], desc.shape[2])

# reshape and restructure labels data in order to corrilate with the descriptors
o = np.ones((1, desc.shape[1]))
y = labelNums.reshape(labelNums.shape[0], 1) @ o
y = y.reshape(-1)

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.8, test_size=0.2)


# In[11]:


from sklearn.cluster import KMeans, MiniBatchKMeans
n_clust = 128

kmeans = MiniBatchKMeans(n_clusters=n_clust, batch_size=1000, verbose=True)
get_ipython().run_line_magic('time', 'kmeans.fit(x_train)')


# In[12]:


histograms = []

# For every decriptive image create a histogram and add to the histograms dataset
for image in desc:
    # predict cluster for each image descriptor
    clusters = kmeans.predict(image)
    
    #create histogram for associated clusters to descriptors
    hist, _ = np.histogram(clusters, range(n_clust+1))
    histograms.append(hist)

histograms = np.array(histograms)


# In[53]:


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

scaler = StandardScaler()
plt.figure(figsize=(12,12))
roc_auc_results = []
splits = [ 0.9, 0.8, 0.7, 0.6 ]
Cs = [0.001, 0.02, 1, 100, 1000]
classes = [1,2,3,4,5,6,7,8]

# Binarize the target class to be proper work with OneVsAll method
y = label_binarize(labelNums, classes=classes)
n_classes = y.shape[1]

subplot = 1
lw = 2
plt.figure(figsize=(32,24))

# Test for different split sizes
for split in splits:
    h_train, h_test, y_train, y_test = train_test_split(histograms, y, random_state=0, train_size=split, test_size=(1-split))
    X_train_scaled = scaler.fit_transform(h_train)
    X_test_scaled = scaler.transform(h_test)
    
    # Test different C parameters
    for C in Cs:
        svc = OneVsRestClassifier(SVC(kernel='linear', C=C, probability=True, random_state=0, max_iter=1000))
        svc.fit(X_train_scaled, y_train)
        
        y_score = svc.decision_function(X_test_scaled)
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Add Area under curve to the array with the params
        roc_auc_results.append({'roc_auc': roc_auc[2], 'C': C, 'split': split})
        
        plt.subplot(len(splits),len(Cs),subplot)
        subplot += 1
        plt.plot(fpr[2], tpr[2], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.4f)' % roc_auc[2])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC - C = {} & train size = {}%'.format(C, split))
        plt.legend(loc="lower right")
plt.show()


# In[55]:


import pandas as pd

df = pd.DataFrame(roc_auc_results)
plt.figure(figsize=(12,9))
plt.plot(df.groupby('C').agg(np.mean).index, df.groupby('C').agg(np.mean)['roc_auc'])
plt.xlabel('C')
plt.ylabel('ROC AUC score')
plt.title('BOW Linear SVM classifier ROC AUC score mean for split sizes over C')
plt.xscale("log")


# In[70]:


d = df[df['split']==0.8]

# Calculate max roc auc score
max_roc_auc = np.max(d['roc_auc'])
max_C = d[d['roc_auc'] == max_roc_auc]['C'].values[0]

# Calculate mean roc auc score
mean_roc_auc = np.mean(d['roc_auc'])

print("80%/20% split, The C parameter that maximize the roc auc score is: {}".format(max_C))
print("80%/20% split,  The mean of ROC AUC score for all the different C parameters is: {}".format(mean_roc_auc))


# # Part 2 Q2

# In[71]:


from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.layers import Input
input_tensor = Input(shape=(None, None, 3))
model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
print(model.summary())


# In[93]:


ims = np.array(images)
ims = preprocess_input(ims)

# Predict descriptors
pred = model.predict(ims)


# In[104]:


pred.reshape(pred.shape[0]*pred.shape[1]*pred.shape[2], pred.shape[3]).shape


# In[141]:


pred.reshape(pred.shape[0], pred.shape[1]*pred.shape[2], pred.shape[3]).shape


# In[135]:


# Layout descriptors over 2D matrix
descriptors = pred.reshape(pred.shape[0]*pred.shape[1]*pred.shape[2], pred.shape[3])

# Match labels
o = np.ones((1, pred.shape[1] * pred.shape[2]))
y = labelNums.reshape(labelNums.shape[0], 1) @ o
y = y.reshape(-1)

x_train, x_test, y_train, y_test = train_test_split(descriptors, y, random_state=0, train_size=0.8, test_size=0.2)


# In[136]:


n_clust = 128

kmeans_tnsr = MiniBatchKMeans(n_clusters=n_clust, batch_size=1000, verbose=True)
get_ipython().run_line_magic('time', 'kmeans_tnsr.fit(x_train)')


# In[142]:


histograms_tnsr = []

# For every decriptive image create a histogram and add to the histograms dataset
for image in pred.reshape(pred.shape[0], pred.shape[1]*pred.shape[2], pred.shape[3]):
    # predict cluster for each image descriptor
    clusters = kmeans_tnsr.predict(image)
    
    #create histogram for associated clusters to descriptors
    hist, _ = np.histogram(clusters, range(n_clust+1))
    histograms_tnsr.append(hist)

histograms_tnsr = np.array(histograms_tnsr)


# In[151]:


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

scaler = StandardScaler()
plt.figure(figsize=(12,12))
roc_auc_results = []
splits = [ 0.9, 0.8, 0.7, 0.6 ]
Cs = [0.001, 0.02, 1, 100, 1000]
classes = [1,2,3,4,5,6,7,8]

# Binarize the target class to be proper work with OneVsAll method
y = label_binarize(labelNums, classes=classes)
n_classes = y.shape[1]

subplot = 1
lw = 2
plt.figure(figsize=(32,24))

# Test for different split sizes
for split in splits:
    h_train, h_test, y_train, y_test = train_test_split(histograms_tnsr, y, random_state=0, train_size=split, test_size=(1-split))
    X_train_scaled = scaler.fit_transform(h_train)
    X_test_scaled = scaler.transform(h_test)
    
    # Test different C parameters
    for C in Cs:
        svc = OneVsRestClassifier(SVC(kernel='linear', C=C, probability=True, random_state=0, max_iter=1000))
        svc.fit(X_train_scaled, y_train)
        
        y_score = svc.decision_function(X_test_scaled)
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Add Area under curve to the array with the params
        roc_auc_results.append({'roc_auc': roc_auc[2], 'C': C, 'split': split})
        
        plt.subplot(len(splits),len(Cs),subplot)
        subplot += 1
        plt.plot(fpr[2], tpr[2], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.4f)' % roc_auc[2])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC - C = {} & train size = {}%'.format(C, split))
        plt.legend(loc="lower right")
plt.show()


# In[152]:


import pandas as pd

df = pd.DataFrame(roc_auc_results)
plt.figure(figsize=(12,9))
plt.plot(df.groupby('C').agg(np.mean).index, df.groupby('C').agg(np.mean)['roc_auc'])
plt.xlabel('C')
plt.ylabel('ROC AUC score')
plt.title('BOW Linear SVM classifier ROC AUC score mean for split sizes over C')
plt.xscale("log")


# In[154]:


# Only the 80/20 split
d = df[df['split']==0.8]

# Calculate max roc auc score
max_roc_auc = np.max(d['roc_auc'])
max_C = d[d['roc_auc'] == max_roc_auc]['C'].values[0]

# Calculate mean roc auc score
mean_roc_auc = np.mean(d['roc_auc'])

print("80%/20% split, The C parameter that maximize the roc auc score is: {}".format(max_C))
print("80%/20% split,  The mean of ROC AUC score for all the different C parameters is: {}".format(mean_roc_auc))

