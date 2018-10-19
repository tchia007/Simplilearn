''' 
This code has been copied from the simplilearn tensorflow tutorial 
https://www.youtube.com/watch?v=E8n_k6HNAgs&index=24&list=PLEiEAq2VkUULYYgj13YHUWmRePqiu8Ddy 
'''

#--------------1. Read the census_data.csv using pandas library
import pandas as pandas
census = pd.read_csv("census_data.csv")

#--------------2. Display the head of the dataset 
print census.head()

#--------------3. Convert the label column to 0s and 1s instead of strings 
print census['income_bracket'].unique()

def label_fix(label):
	if label == ' <=50k':
		return 0
	else:
		return 1

census['income_bracket'] = census['income_bracket'].apply(label_fix)

#--------------4. Perform a Train Test split on the dataset
from sklearn.model_selection import train_test_split

x_data = census.drop('income_bracket', axis = 1)
y_labels = census['income_bracket']
X_train, X_test, y_train, y_test = train_test_split(x_data, y_labels, test_size = 0.3, random_state = 101)

#--------------5. Create the feature columns for the categorical values using library lists or hash bucks
import tensorflow as tf 

#Create the tf.feature_columns for the categorical values. use the vocabulary lists or just use hash buckets 
#vocublary list is when we know the exact values that can be taken. look at gender
#hash bucket can have any value 
#hash_bucket_size is how many values that the category can be. usually try to pick larger value
gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ["Female", "Male"]) 
occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital_status", hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native_country", hash_bucket_size=1000)

#--------------6. Create the feature columns for the continiuous values using numeric column 
#Create the continuous feature_columns for the continuous values using numeric_column 
age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")

#--------------7. Put all these variables into a single list with the variable name feat_cols
feat_cols = [gender, occupation, marital_status, relationship, education, workclass, native_country,
			 age, education_num, capital_gain, capital_loss, hours_per_week]

#--------------8. Creat the input function. Batch size is up to you
input_func = tf.estimator.inputs.pandas_input_fn(x = X_train, y = y_train, batch_size = 100, num_epochs = None, shuffle = True)

#--------------9. Create the model with tf.estimator using LinearClassifier 
model = tf.estimator.LinearClassifier(feature_columns = feature_cols)

#--------------10. Train the model for at least 5000 steps
model.train(input_fn = input_func, steps = 5000)

#--------------11. Evaluation fo the model
pred_fn = tf.estimator.inputs.pandas_input_fn(x = X_test, batch_size = len(X_test), shuffle = False)

#--------------12. Create a list of class_ids key values from the prediction list dictionaries. 
##-----------------These predictions will be used to compare against y_test values 
final_preds = [] 
for pred in predictions:
	final_preds.append(pred['class_ids'][0])

print final_preds[:10]

#--------------13. Calculat the models performance on Test data
from sklearn.metrics import classification_report
print classification_report(y_test, final_preds)














