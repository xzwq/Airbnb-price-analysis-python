import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, roc_curve, auc, roc_auc_score, recall_score, mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error,root_mean_squared_error
import glob

all_data = pd.DataFrame()
for f in glob.glob("C:\\Users\\ege42\\OneDrive\\Masaüstü\\yz proje\\dataset\\*.csv"):
    df = pd.read_csv(f, sep=",")
    all_data = pd.concat([all_data, df], ignore_index=True)

imputer = SimpleImputer(strategy="mean")
all_data.fillna(value="Missing", inplace=True)

non_numeric_columns = all_data.select_dtypes(include=['object']).columns
label_encoders = {}
for col in non_numeric_columns:
    if all_data[col].nunique() > 1:
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col].fillna('Missing').astype(str))
        label_encoders[col] = le

float_cols = all_data.select_dtypes(include=['float64']).columns
all_data[float_cols] = all_data[float_cols].round(2)

bool_cols = all_data.select_dtypes(include=['bool']).columns
all_data[bool_cols] = all_data[bool_cols].astype(int)

all_data.rename(columns={
    'realSum': 'Kira Fiyatı',
    'room_type': 'Oda Tipi',
    'room_shared': 'Oda paylaşımlı mı?',
    'room_private': 'Oda özel mi?',
    'host_is_superhost': 'Ev Sahibi Süper Ev Sahibi mi?',
    'multi': 'Girişin birden fazla oda için olup olmadığı',
    'biz': 'İlanın ticari amaçlı olup olmadığı',
    'guest_satisfaction_overall': 'Misafir Memnuniyeti',
    'person_capacity': 'Kişi Kapasitesi',
    'cleanliness_rating': 'Temizlik Puanı',
    'bedrooms': 'Yatak Odası Sayısı',
    'dist': 'Şehir Merkezine Uzaklık',
    'metro_dist': 'Metroya Uzaklık',
    'attr_index': 'Çekicilik Endeksi',
    'attr_index_norm': 'Normalleştirilmiş Çekicilik Endeksi',
    'lat': 'Enlem',
    'lng': 'Boylam'
}, inplace=True)


print("=================Linear Regression=======================")

all_data['Kira Fiyatı Log'] = np.log10(all_data['Kira Fiyatı'])

a_log = all_data[[özellik for özellik in all_data.columns if özellik != 'Kira Fiyatı' and özellik != 'Kira Fiyatı Log']]
b_log = all_data['Kira Fiyatı Log']

a_train_log, a_test_log, b_train_log, b_test_log = train_test_split(a_log, b_log, test_size=0.1, random_state=50)

model_log = LinearRegression()
model_log.fit(a_train_log, b_train_log)

b_pred_log = model_log.predict(a_test_log)

mse_log = mean_squared_error(b_test_log, b_pred_log)
r2_log = r2_score(b_test_log, b_pred_log)
mae_log = mean_absolute_error(b_test_log, b_pred_log)
rmse = root_mean_squared_error(b_test_log, b_pred_log)
mape = mean_absolute_percentage_error(b_test_log, b_pred_log)   

print(f"MSE: {int(mse_log*100)}")
print(f"R2: {r2_log*100:.2f}")
print(f"RMSE: {int(rmse*100)}")
print(f"MAE: {int(mae_log*100)}")
print(f"MAPE: {int(mape*100)}")


plt.figure(figsize=(10, 6))
plt.scatter(b_test_log, b_pred_log, color='blue', edgecolor='k', alpha=0.7)
plt.plot([min(b_test_log), max(b_test_log)], [min(b_test_log), max(b_test_log)], color='red', linewidth=2)
plt.xlabel('Gerçek Kira Fiyatı Log')
plt.ylabel('Tahmin Edilen Kira Fiyatı Log')
plt.title('Linear Regression: Gerçek vs Tahmin Edilen Kira Fiyatı Log')
plt.show()

print("======================K-Means============================")


fiyatlar = all_data[['Kira Fiyatı']].values

km = KMeans(n_clusters=3, init='k-means++', random_state=42)
kume_etiketleri = km.fit_predict(fiyatlar)

kume_siralama = np.argsort(km.cluster_centers_.flatten())
kategori_mapping = {kume_siralama[0]: 'Düşük', 
                    kume_siralama[1]: 'Orta', 
                    kume_siralama[2]: 'Yüksek'}

kategori_etiketleri = [kategori_mapping[label] for label in kume_etiketleri]

all_data['Fiyat Kategorisi'] = kategori_etiketleri


y_true = all_data['Fiyat Kategorisi']
y_pred = kategori_etiketleri


y_true_bin = label_binarize(y_true, classes=['Düşük', 'Orta', 'Yüksek'])
y_pred_bin = label_binarize(y_pred, classes=['Düşük', 'Orta', 'Yüksek'])


accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.2f}")


f_measure = f1_score(y_true, y_pred, average='weighted')
print(f"F-measure: {f_measure:.2f}")


recall = recall_score(y_true, y_pred, average='weighted')
print(f"Recall: {recall:.2f}")


precision = precision_score(y_true, y_pred, average='weighted')
print(f"Precision: {precision:.2f}")


roc_auc = roc_auc_score(y_true_bin, y_pred_bin, average='weighted', multi_class='ovr')
print(f"AUC: {roc_auc:.2f}")

specifity = confusion_matrix(y_true, y_pred)[0,0]/(confusion_matrix(y_true, y_pred)[0,0]+confusion_matrix(y_true, y_pred)[0,1])
print(f"Specificity: {specifity:.2f}")



fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


plt.figure()
for i in range(3):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve (area = {roc_auc[i]:.2f}) for class {i}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for K-Means')
plt.legend(loc="lower right")
plt.show()

print("====================Decision Tree==========================")
all_data['Kira Fiyatı Kategorik'] = all_data['Kira Fiyatı'].apply(lambda x: 'Düşük' if x < 300 else 'Yüksek')

feature_cols = [
    'Oda Tipi', 'Oda paylaşımlı mı?', 'Oda özel mi?', 'Ev Sahibi Süper Ev Sahibi mi?',
    'Girişin birden fazla oda için olup olmadığı', 'İlanın ticari amaçlı olup olmadığı',
    'Misafir Memnuniyeti', 'Kişi Kapasitesi', 'Temizlik Puanı',
    'Yatak Odası Sayısı', 'Şehir Merkezine Uzaklık', 'Metroya Uzaklık',
    'Çekicilik Endeksi', 'Normalleştirilmiş Çekicilik Endeksi', 'Enlem', 'Boylam'
]

x = all_data[feature_cols]
y = all_data['Kira Fiyatı Kategorik']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=45,shuffle=True)

dt = DecisionTreeClassifier()
dt = dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

print(f"Accuracy DT: {metrics.accuracy_score(y_test, y_pred)*100:.2f}")
print(f"F1 Score DT: {f1_score(y_test, y_pred, average='weighted')*100:.2f}")
print(f"Recall DT: {recall_score(y_test, y_pred, average='weighted')*100:.2f}")
print(f"Precision DT: {precision_score(y_test, y_pred, average='weighted')*100:.2f}")
print(f"Confusion Matrix DT: {confusion_matrix(y_test, y_pred)}")
specifity_dt = confusion_matrix(y_test, y_pred)[0,0]/(confusion_matrix(y_test, y_pred)[0,0]+confusion_matrix(y_test, y_pred)[0,1])
print(f"Specificity DT: {specifity_dt*100:.2f}")


print("=========================KNN===============================")
etiketkümesi = all_data['Kira Fiyatı Kategorik']
ozellikkümesi = all_data[feature_cols]

ozellikkümesi = pd.DataFrame(imputer.fit_transform(ozellikkümesi))

ozellikkümesi_train, ozellikkümesi_test, etiketkümesi_train, etiketkümesi_test = train_test_split(
    ozellikkümesi, etiketkümesi, test_size=0.1, random_state=10
)

knn = KNeighborsClassifier(n_neighbors=15, metric='hamming', algorithm='auto', weights='distance')
knn.fit(ozellikkümesi_train, etiketkümesi_train)

tahminsonuc = knn.predict(ozellikkümesi_test)
confusion_mat = confusion_matrix(etiketkümesi_test, tahminsonuc)

accuracy = accuracy_score(etiketkümesi_test, tahminsonuc)
f1 = f1_score(etiketkümesi_test, tahminsonuc, average='weighted')
recall = recall_score(etiketkümesi_test, tahminsonuc, average='weighted')
precision = precision_score(etiketkümesi_test, tahminsonuc, average='weighted')


print("Confusion Matrix KNN:")
print(confusion_mat)
print(f"Doğruluk KNN: {accuracy*100:.2f}")
print(f"F1 Skoru KNN: {f1*100:.2f}")
print(f"Recall KNN: {recall*100:.2f}")
print(f"Precision KNN: {precision*100:.2f}")
specifity = confusion_mat[0,0]/(confusion_mat[0,0]+confusion_mat[0,1])
print(f"Specificity KNN: {specifity*100:.2f}")




print("=====================Naive Bayes===========================")


X_nb = all_data.iloc[:, 2:20]
y_nb = all_data.iloc[:, -1]

X_nb = pd.DataFrame(imputer.fit_transform(X_nb))

X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X_nb, y_nb, test_size=0.1, random_state=42)

nb_model = GaussianNB()
nb_model.fit(X_train_nb, y_train_nb)

y_pred_nb = nb_model.predict(X_test_nb)



accuracy_nb = accuracy_score(y_test_nb, y_pred_nb)


print(f"Doğruluk Naive Bayes: {int(accuracy_nb*100)}")
print(f"F1 Skoru Naive Bayes: {f1_score(y_test_nb, y_pred_nb, average='weighted')*100:.2f}")
print(f"Recall Naive Bayes: {recall_score(y_test_nb, y_pred_nb, average='weighted')*100:.2f}")
print(f"Precision Naive Bayes: {precision_score(y_test_nb, y_pred_nb, average='weighted')*100:.2f}")
print(f"Confusion Matrix Naive Bayes: {confusion_matrix(y_test_nb, y_pred_nb)}")
specifity_nb = confusion_matrix(y_test_nb, y_pred_nb)[0,0]/(confusion_matrix(y_test_nb, y_pred_nb)[0,0]+confusion_matrix(y_test_nb, y_pred_nb)[0,1])
print(f"Specificity Naive Bayes: {specifity_nb*100:.2f}")


print("=================SVM=======================")

price_bins = [0, 100, 200, 300, 400, 500, np.inf]
price_labels = ['0-100', '100-200', '200-300', '300-400', '400-500', '500+']
all_data['Kira Fiyatı Kategorik'] = pd.cut(all_data['Kira Fiyatı'], bins=price_bins, labels=price_labels)

etiket_svm = all_data['Kira Fiyatı Kategorik']
ozellik_svm = [
    'Oda Tipi', 'Oda paylaşımlı mı?', 'Oda özel mi?', 'Ev Sahibi Süper Ev Sahibi mi?',
    'Misafir Memnuniyeti', 'Kişi Kapasitesi', 'Temizlik Puanı',
    'Yatak Odası Sayısı', 'Şehir Merkezine Uzaklık', 'Metroya Uzaklık'
]

all_data_sample = all_data.sample(frac=1, random_state=42)

x_sample = all_data_sample[ozellik_svm]
y_sample = all_data_sample['Kira Fiyatı Kategorik']

x_train_svm, x_test_svm, y_train_svm, y_test_svm = train_test_split(x_sample, y_sample, test_size=0.1, random_state=50)

svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(x_train_svm, y_train_svm)

y_pred_svm = svm_model.predict(x_test_svm)

accuracy_svm = accuracy_score(y_test_svm, y_pred_svm)
f1_svm = f1_score(y_test_svm, y_pred_svm, average='weighted')
recall_svm = recall_score(y_test_svm, y_pred_svm, average='weighted')
precision_svm = precision_score(y_test_svm, y_pred_svm, average='weighted',zero_division=0)

conf_mat_svm = confusion_matrix(y_test_svm, y_pred_svm)
print("Confusion Matrix SVM:")
print(conf_mat_svm)

print(f"Accuracy SVM: {accuracy_svm*100:.2f}")
print(f"F1 Score SVM: {f1_svm*100:.2f}")
print(f"Recall SVM: {recall_svm*100:.2f}")
print(f"Precision SVM: {precision_svm*100:.2f}")
print(f"Sensitivity SVM: {recall_svm*100:.2f}")

