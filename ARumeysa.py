import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv("ARumeysa.csv")

X = df.drop("sinif", axis=1)
y = df["sinif"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

etiketler = ["Yaş", "Kahve Sayısı", "Dizi Süresi", "Kitap Sayısı"]
kurallar = export_text(model, feature_names=etiketler)
print("📘 Karar Ağacı Kuralları:\n")
print(kurallar)

y_pred = model.predict(X_test)

print("\n📊 Karışıklık Matrisi:")
print(confusion_matrix(y_test, y_pred))

class_names_str = [str(cls) for cls in model.classes_]  # int değerler varsa string'e çevir
print("\n📋 Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=class_names_str))

plt.figure(figsize=(12, 8))
plot_tree(
    model,
    feature_names=etiketler,
    class_names=class_names_str,
    filled=True,
    rounded=True
)
plt.title("Karar Ağacı Görselleştirmesi")
plt.savefig("karar_agaci.png")  # PNG dosyasına kaydet
plt.show()  # Ekranda göster
