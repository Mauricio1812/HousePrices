from sklearn.preprocessing import LabelEncoder
import json 

lista = ["20", "30", "40", "45", "50", "60", "70", "75", "80", "85", "90", "120", "150", "160", "180", "190",]        

label = LabelEncoder() 
label.fit(lista) 
lista = label.transform(lista)

#lista = label.fit_transform(lista)
total = dict(zip(label.classes_, label.transform(label.classes_)))

print(lista)

#Escribiendo el arreglo en un archivo de texto
file = open('Diccionario.txt', "a")
file.write(str(total))
file.write(" ")
file.close()

