import numpy as np
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
data=pd.read_csv("arrhythmia.data", header=None, na_values="?")
data.dropna(inplace=True)

X=data.iloc[:,:-1].values #atributele esantioanelor
Y=data.iloc[:,-1].values #etichetele esantioanelor
# 2. Reformulare etichete: 1 -> 0 (normal), 2–16 -> 1 (aritmie)D:\Proiect_Inteligenta_Artificiala
Y_binary = np.where(Y == 1, 0, 1)
#Scalam datele de la intrare, intrucat avem foarte multe esantioane cu eticheta 1 si mai putine cu alte etichete
scaler=StandardScaler()
X_scalat=scaler.fit_transform(X)
#Antrenam modelul SVM
X_train, X_test, Y_train, Y_test= train_test_split(X_scalat,Y_binary,test_size=0.25, random_state=42) #Impartim in date si etichete 75 % si 25%

#Incercam sa testam pentru diverse valori de cost si gamma
Cost=[2**-5, 2**-3, 2**-1,1, 2,3,4,5, 2**3, 2**5, 2**7]
gamm=[2**-15, 2**-13, 2**-11, 2**-9, 2**-7, 2**-5, 2**-3, 2**-1, 1,2**1, 3,4,5,2**3,2**4]
#Cream un vector care sa stocheze valorile datelor de acuratete
acuratete=[]
for i in Cost:
    for j in gamm:
        #Testam pentru fiecare cost si gamma din vectori
        svc=svm.SVC(kernel='rbf', C=i,gamma=j, class_weight='balanced')
        svc.fit(X_train,Y_train)

        #Facem predictie
        predictie=svc.predict(X_test)
        acc=accuracy_score(Y_test,predictie)
        print(f"Acuratetea modelului SVM: {acc:.4f} pentru C={i} si gamma={j}")
        acuratete.append(acc)
        if(acc==max(acuratete)):
            cost_optim=i
            gamm_optim=j
print("Acuratetea maxima ",max(acuratete),"obtinut pentru costul",cost_optim,"si pentru parametrul gamma",gamm_optim)