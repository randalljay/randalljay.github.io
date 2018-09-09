---
layout: post
title: Client Project
---





![_config.yml]({{ site.baseurl }}/images/config.png)


# 1.0 Title : Predict factors indicating post survey completion for Uprise customers.


## Datasets

### 1. Uprise dataset

### 2. Intercom dataset (Merged both on email_id)


# 2.0 Data Import , Cleaning and Exploratory Data Analysis


In [209]:

In [210]:

In [211]:

```
df_raw=pd.read_csv('./datasets/rawdf.csv')
```
```
df_raw.shape
```
```
df_raw[['Age_Group','Gender','Industry']].head()
```
Out[210]:

#### (1176, 230)

Out[211]:

### Age_Group Gender Industry

### 0 NaN NaN NaN

### 1 NaN NaN NaN

### 2 39 1.0 1

### 3 22 1.0 9

### 4 27 1.0 3


## 2.1 Data Cleaning, Processing and Feature Engineering:

### This was an interactive process.

### 1. Changed dtypes

### 2. Used Regex to remove unwanted characters

### 3. Created a Data Dictionary and renamed columns.

### 4. Removed rows with null values (Business input).

### 5. Convert all categorical columns to numeric/discrete columns.

### 6. Removed outliers.

### 7. Use Tf-IDF to nd most common industries and narrow down number of

### industries from 1000 categories’ of industries to 14 major ones.

### 8. Used 'Pandas Proling' library to lter the columns with lowest information gain.

### 9. Convert Target variable 'post_psycap_total' from numeric to discrete.

### a) Row with numeric value in 'post_psycap_total' labelled as 1.a) Row with numeric value in 'post_psycap_total' labelled as 1.

### b) Row who Null value in 'post_psycap_total' labelled as 0.b) Row who Null value in 'post_psycap_total' labelled as 0.


In [212]:

In [213]:

In [214]:

```
df_clean=pd.read_csv('./datasets/cleandf_new.csv')
```
```
df_clean[['age','Gender','industry']].head()
```
```
df_clean.shape
```
Out[213]:

### age Gender industry

### 0 39.0 female nance accounting

### 1 22.0 female other

### 2 27.0 female engineering technology

### 3 23.0 female nance accounting

### 4 42.0 male other

Out[214]:

#### (972, 167)


In [215]:
df_clean[['pre_psycap_total','post_psycap_total']].describe().T

Out[215]:

### count mean std min 25% 50% 75% m

### pre_psycap_total 447.0 67.080537 16.197707 10.0 60.0 70.0 80.0 10

### post_psycap_total 162.0 76.728395 13.296908 30.0 70.0 77.5 85.0 10


# 2.2 Exploratory Data Analysis

## 1. Explore the entire data set to nd trends.

## 2. Univariate Analysis

## 3. Boxplot visualisation.

## 4. Remove outliers.

## 5. Histogram visualisation.

## 6. Interesting insights visualising dates and time.


## 2.2.1 Univariate Analysis

In [216]:

In [217]:

```
df_eda=pd.read_csv('./datasets/eda_csv.csv')
```
```
plt.hist(df_eda['age'],label='Age',bins= 20 )
```
```
None
```

In [218]:
_## Outlier detection_

```
sns.boxplot('pre_doc_visits', data=df_eda)
```
```
plt.tick_params(axis='both', labelsize= 15 )
```

In [219]:
df_eda[['pre_psycap_total','post_psycap_total']].hist()

```
None
```

## 2.2.2 Interesting Insights


In [220]:
_##_

```
ax = sns.stripplot(x="age",y='OS' ,hue="Gender", data=df_eda,size= 7 )
```
```
plt.title('Age & Operating System',fontsize= 15 )
```
```
None
```

In [223]:
df_grouped.plot(kind='bar', x='pre_survey_WD',subplots= **True** ,

```
legend= False )
```
```
None
```

In [224]:
df_grouped_2.plot(kind='bar', x='post_survey_WD',subplots= **True** ,

```
legend= False )
```
```
None
```

In [225]:
_### industry-wise pre and post mean wellness total_

```
df_eda.groupby('industry')['post_well_total','pre_well_total'].mean().plot(kind='b
```
```
arh',figsize = ( 16 , 8 ))
```
```
plt.xlabel('Wellness level average',fontsize= 20 )
```
```
plt.ylabel('Industry',fontsize= 20 )
```
```
plt.title('Industry pre/post wellness level',fontsize= 20 )
```
```
plt.legend(loc='upper right', bbox_to_anchor=(1., 0.7),fontsize= 15 )
```
```
plt.tick_params(axis='both', labelsize= 15 )
```
```
None
```

In [226]:
_### industry-wise pre and post mean stress total_

```
df_eda.groupby('industry')['post_stress4_total','pre_stress4_total'].mean().plot(
```
```
kind='barh',figsize = ( 16 , 8 ))
```
```
plt.xlabel('Stress level average',fontsize= 20 )
```
```
plt.ylabel('Industry',fontsize= 20 )
```
```
plt.title('Industry pre/post stress level',fontsize= 20 )
```
```
plt.legend(loc='upper right', bbox_to_anchor=(1., 0.62),fontsize= 15 )
```
```
plt.tick_params(axis='both', labelsize= 15 )
```
```
None
```

In [227]:
plt.figure(figsize=( 15 , 8 ))

```
sns.countplot(x="last_opened_email_WD", hue="Gender", data=df_eda, palette="plasma
```
```
_r")
```
```
plt.tick_params(axis='both', labelsize= 15 )
```
```
plt.ylabel('Count',fontsize= 20 )
```
```
plt.xlabel('Last Opened email day',fontsize= 20 )
```
```
plt.title('Gender last opened email day',fontsize= 20 )
```
```
plt.legend(loc='upper right',fontsize= 15 )
```
```
None
```

In [228]:
plt.figure(figsize=( 15 , 8 ))

```
sns.countplot(x="h_last_opened_email", hue="Gender", data=df_eda, palette="plasma_
```
```
r")
```
```
plt.tick_params(axis='both', labelsize= 15 )
```
```
plt.legend(loc='upper right',fontsize= 15 )
```
```
plt.ylabel('Count',fontsize= 20 )
```
```
plt.xlabel('Last Opened email Hour',fontsize= 20 )
```
```
plt.title('Gender last opened email hour',fontsize= 20 )
```
```
None
```

In [229]:
df_clean.groupby('Gender')['survey_preprogram',

```
'video_retraining','video_pleasant',
```
```
'video_letgo','survey_postprogram'].count(
```
```
).plot(kind='bar',figsize=( 6 , 4 ))
```
```
plt.ylabel('Number of Partipants',fontsize= 15 )
```
```
plt.xlabel('Gender',fontsize= 15 )
```
```
plt.title('Reducing participation',fontsize= 15 )
```
```
plt.legend(loc= 0 ,fontsize= 10 )
```
```
plt.tick_params(axis='both', labelsize= 15 )
```

# 2.3 Modelling and Machine Learning


### 2.3.1 Correlation and Feature selection

### 1. Drop all post survey features from dataframe

### 2. Variance Inflation Factor

### 3. Correlation Heatmap


In [233]:
vif.sort_values(by='VIF_Factor',ascending= **True** ).head( 10 )

Out[233]:

### VIF_Factor features

### 34 1.191108 pre_diff_not_paid

### 19 1.231367 bonus_resources_retraining

### 23 1.252392 web_sess_total

### 29 1.272511 pre_diff_unclear_role

### 0 1.298058 age

### 35 1.314345 pre_diff_lacking_support

### 18 1.318270 video_retraining

### 31 1.372747 pre_diff_job_uncertainty

### 32 1.387862 pre_diff_lack_resources

### 30 1.389767 pre_diff_not_recognised


In [235]:
sns.set(style="white")

```
#make correlation matrix
```
```
corr = X.corr()
```
```
# Mask the upper half
```
```
mask = np.zeros_like(corr, dtype=np.bool)
```
```
mask[np.triu_indices_from(mask)] = True
```
```
# assign plt size
```
```
f, ax = plt.subplots(figsize=( 11 , 9 ))
```
```
cmap = sns.diverging_palette( 220 , 10 , as_cmap= True )
```
```
# Call the heatmap with the mask and correct aspect ratio
```
```
sns.heatmap(corr, mask=mask, cmap=cmap, vmax= 1 , center= 0 ,
```
```
square= True , linewidths=. 5 , cbar_kws={"shrink":. 5 })
```
Out[235]:

```
<matplotlib.axes._subplots.AxesSubplot at 0x1a28f7efd0>
```


### 2.3.2 Sentiment Analysis (Customer Reviews)


In [238]:
_## Top five reviews based on sentiment of the reviews_

```
df_sent.head()
```
Out[238]:

### pre_stress4_total post_stress4_total up_one_sentence compound

### 81 6.0 6.0

### Amazing opportunity

### to do something

### positive f...

### 0.9360

### 890 6.0 31.0

### This is a great program

### to start discussing an...

### 0.9325

### 418 44.0 19.0

### The Uprise program

### has been great. Liza

### was re...

### 0.8923

### 795 50.0 31.0

### Supportive, offers

### great tools, and helps

### to s...

### 0.8910

### 448 50.0 50.0

### A good program to

### prompt action for

### better wel...

### 0.8832


In [88]:
print(wordcloud)

```
fig = plt.figure( 1 )
```
```
plt.imshow(wordcloud)
```
```
plt.axis('off')
```
```
plt.show()
```
```
fig.savefig("word1.png", dpi= 900 )
```
```
None
```
```
<wordcloud.wordcloud.WordCloud object at 0x1a303ffc88>
```

### 2.3.3 Application of Machine Learning Algorithims

### Target Variable : Predict factors for post survey completion for Uprise customers

### 1. Up scaling imbalanced data set

### 2. Multiple model comparison

### 3. AUC ROC


In [247]:
_#note the problem of imbalanced target variable_

```
X_train.complete_post_survey.value_counts()
```
Out[247]:

#### 0 209

#### 1 103

```
Name: complete_post_survey, dtype: int64
```

In [248]:
_# Separate majority and minority classes_

```
df_majority = X_train[X_train.complete_post_survey== 0 ]
```
```
df_minority = X_train[X_train.complete_post_survey== 1 ]
```
```
# Upsample minority class
```
```
df_minority_upsampled = resample(df_minority,
```
```
replace= True , # sample with replacement
```
```
n_samples= 209 , # to match majority class
```
```
random_state= 123 ) # reproducible results
```
```
# Combine majority class with upsampled minority class
```
```
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
```
```
# Display new class counts
```
```
df_upsampled.complete_post_survey.value_counts()
```
Out[248]:

#### 1 209

#### 0 209

```
Name: complete_post_survey, dtype: int64
```

In [272]:
_#sort values by randomforest_

```
f_coe.sort_values(by='Rf_coe',ascending= False ).head( 5 )
```
Out[272]:

### Feature Logit_coe Tree_coe Rf_coe

### 21 video_letgo 1.88 0.66 0.19

### 22 video_breathing 0.82 0.06 0.14

### 23 web_sess_total 0.00 0.08 0.09

### 20 video_pleasant 0.48 0.01 0.08

### 24 total_appointments 1.00 0.05 0.06


In [274]:

## AUC-ROCAUC-ROC

### I used the AUC-ROC curve as a metric to test my models because of the class imbalance in

### my data set. AUC-ROC is not effected by class imbalance however accuracy score is affected.

```
class_summary
```
Out[274]:

### Best Model Score AUCROC Comparison

### Model

### Logistic 0.82 0.71

### KNN 0.72 0.71

### SVM 0.83 0.81

### DecisionTree 0.71 0.86

### RandomForest 0.86 0.70

### Baseline 0.61 0.61


In [275]:
plt.figure(figsize=( 16 , 8 )).clf()

```
#plot decision tree
```
```
pred = t_yhat_pp[:, 1 ]
```
```
label = y_test.values
```
```
fpr, tpr, thresh = metrics.roc_curve(label, pred)
```
```
auc = round(metrics.roc_auc_score(label, pred), 2 )
```
```
plt.plot(fpr,tpr,label="Decision Tree, auc="+str(auc))
```
```
#plot random forest
```
```
pred = r_yhat_pp[:, 1 ]
```
```
label = y_test.values
```
```
fpr, tpr, thresh = metrics.roc_curve(label, pred)
```
```
auc = round(metrics.roc_auc_score(label, pred), 2 )
```
```
plt.plot(fpr,tpr,label="Random Forest, auc="+str(auc))
```
```
#plot SVM
```
```
pred = s_yhat_pp[:, 1 ]
```
```
label = y_test.values
```
```
fpr, tpr, thresh = metrics.roc_curve(label, pred)
```
```
auc = round(metrics.roc_auc_score(label, pred), 2 )
```
```
plt.plot(fpr,tpr,label="SVM, auc="+str(auc))
```
```
#plot knn
```
```
pred = k_yhat_pp[:, 1 ]
```
```
label = y_test.values
```
```
fpr, tpr, thresh = metrics.roc_curve(label, pred)
```
```
auc = round(metrics.roc_auc_score(label, pred), 2 )
```
```
plt.plot(fpr,tpr,label="KNN, auc="+str(auc))
```
```
#plot logistic
```
```
pred = l_yhat_pp[:, 1 ]
```
```
label = y_test.values
```
```
fpr, tpr, thresh = metrics.roc_curve(label, pred)
```
```
auc = round(metrics.roc_auc_score(label, pred), 2 )
```
```
plt.plot(fpr,tpr,label="Logistic, auc="+str(auc))
```

_#plot random predictor_

plt.plot([ 0 , 1 ],[ 0 , 1 ],'r--')

plt.tick_params(axis='both', labelsize= 20 )

plt.ylabel('True Positive Rate',fontsize= 20 )

plt.xlabel('False Positive Rate',fontsize= 20 )

plt.title('Area under the ROC Curve.',fontsize= 20 )

plt.legend(loc= 0 ,fontsize= 15 )

**None**
