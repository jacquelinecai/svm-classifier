import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import csv
import librosa

with open('train.csv', mode='r') as csv_file:
  reader = csv.reader(csv_file)
  training_labels = [row for row in reader]

# [extract_features(fp)] extracts mfccs, chroma, and zero_crossing_rate features 
# from the librosa package
def extract_features(filepath):
  y, sr = librosa.load(filepath)
  mfccs = librosa.feature.mfcc(y=y, sr=sr)
  chroma = librosa.feature.chroma_stft(y=y, sr=sr)
  zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
  return np.hstack((mfccs.flatten(), chroma.flatten(), zero_crossing_rate.flatten()))

# relevant training and testing sets
X_train = []
Y_train = []

# represents the sets for choosing a parameter C
X_ctrain = []
Y_ctrain = []
X_ctest = []
Y_ctest = []

# retrieves the audio files and extracts features and labels
for file in range(48000):
  filepath = "train/train_" + str(file) + ".wav"
  features = extract_features(filepath)
  X_train.append(features)
  label = training_labels[file+1][1]
  Y_train.append(label)
  if file % 5 == 0:
    X_ctrain.append(features)
    Y_ctrain.append(label)
  if file % 5 == 3:
    X_ctest.append(features)
    Y_ctest.append(label)

# splitting the training set into 2 parts
X_train_split, X_test_split, Y_train_split, Y_test_split = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

X_train_split = np.array(X_train_split)
Y_train_split = np.array(Y_train_split)
X_test_split = np.array(X_test_split)
Y_test_split = np.array(Y_test_split)
X_ctrain = np.array(X_ctrain)
Y_ctrain = np.array(Y_ctrain)
X_ctest = np.array(X_ctest)
Y_ctest = np.array(Y_ctest)

# normalizing the data
mean_value = np.mean(X_train_split, axis=0)
std_dev = np.std(X_train_split, axis=0)

X_train_split = (X_train_split - mean_value) / std_dev
X_test_split = (X_test_split - mean_value) / std_dev
X_ctrain = (X_ctrain - mean_value) / std_dev
X_ctest = (X_ctest - mean_value) / std_dev

# compressing the data down to [n] dimensions
n = min(X_train_split.shape[0], X_train_split.shape[1])
pca = PCA(n_components=n)

X_train_split = pca.fit_transform(X_train_split)
X_test_split = pca.transform(X_test_split)
X_ctrain = pca.transform(X_ctrain)
X_ctest = pca.transform(X_ctest)

max_accuracy = 0
max_c = 1

# training the SVC model on different parameters of C and obtaining the one 
# with the best accuracy
C = [x for x in range(1,21)]
for c in C:
  svc = SVC(C=c)
  svc.fit(X_ctrain, Y_ctrain)
  test_accuracy = svc.score(X_ctest, Y_ctest)
  if test_accuracy > max_accuracy:
    max_accuracy = test_accuracy
    max_c = c

print(max_c)
print(max_accuracy)

# training the SVC model on the split training set
svc = SVC(C=max_c)
svc.fit(X_train_split, Y_train_split)
accuracy = svc.score(X_train_split, Y_train_split)

print(accuracy)

predicted_labels = []

# [predict_labels(m, f, mv, sd, pca)] determines the predicted labels for model 
# [m] with features [f], normalization using [mv] and [sd], and compression with 
# [pca]. Returns the predicted label
def predict_labels(model, features, mean_value, std_dev, pca):
  features_standardized = (features - mean_value) / std_dev
  features_standardized = features_standardized.reshape(1, -1)
  features_pca = pca.transform(features_standardized)
  labels_pred = model.predict(features_pca)
  return labels_pred

# predicts the labels of each audio file in the test set
for file in range(48000):
  filepath = "test/test_" + str(file) + ".wav"
  features = extract_features(filepath)
  predicted_label = predict_labels(svc, features, mean_value, std_dev, pca)
  predicted_labels.append((file, predicted_label[0]))

# appends the labels onto a csv file
output_file_path = "predicted_labels.csv"
with open(output_file_path, mode='w', newline='') as csv_file:
  writer = csv.writer(csv_file)
  writer.writerow(['# ID', 'label'])
  writer.writerows(predicted_labels)
