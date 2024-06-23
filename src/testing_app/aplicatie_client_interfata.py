import joblib
import time
import tkinter as tk
from tkinter import ttk, messagebox
import requests
import random
import numpy as np




# Incarcarea datelor de test, pentru a nu influenta timpul de detectie cu supraadaptarea

test_data = joblib.load('test_data.pkl')
X_test = test_data['X_test'].values
y_test = test_data['y_test'].values



# Definirea functiei de trimitere a cererii catre serverul de predictie

def send_request(sample, model):
    url = 'http://127.0.0.1:5000/predict'
    headers = {'Content-Type': 'application/json'}
    data = {
        'features': sample,
        'model': model
    }
    
    # Masurarea timpului total de detectie (timpul de raspuns al serverului + timpul de procesare al modelului selectat)

    start_time = time.time()

    response = requests.post(url, headers=headers, json=data)

    end_time = time.time()
    total_time = end_time - start_time

    return response.json(), total_time



# Aplicatia cu interfata grafica pentru testarea timpului de detectie al IDS-ului

class IDSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Test al timpului de detecție")

        self.root.geometry("600x300")
        
        self.classifier_label = tk.Label(root, text="Selectați un algoritm:")
        self.classifier_label.pack(anchor='center', pady=(80, 5))
        
        self.classifier_var = tk.StringVar(value="Testare fără algoritm")
        self.classifier_menu = ttk.Combobox(root, textvariable=self.classifier_var)
        self.classifier_menu['values'] = ("Testare fără algoritm",
                                            "AdaBoostClassifier",
                                            "BaggingClassifier",
                                            "DecisionTreeClassifier",
                                            "RandomForestClassifier",
                                            "ExtraTreesClassifier",
                                            "GradientBoostingClassifier",
                                            "HistGradientBoostingClassifier")
        self.classifier_menu.pack(anchor='center', pady=5)
        
        self.start_button = tk.Button(root, text="Start Test", command=self.start_test)
        self.start_button.pack(anchor='center', pady=5)
        
        self.result_label_prediction = tk.Label(root, text="")
        self.result_label_prediction.pack(anchor='center', pady=10)

        self.result_label_time = tk.Label(root, text="")
        self.result_label_time.pack(anchor='center', pady=10)

        self.result_label_avg_operations_per_second = tk.Label(root, text="")
        self.result_label_avg_operations_per_second.pack(anchor='center', pady=5)

    def start_test(self):
        classifier = self.classifier_var.get()
        all_times = []
        predictions = []


        # Se vor selecta la intamplare 10 esantioane din setul de date de testare pentru a calcula timpul de detectie ca o medie a fiecaruia

        random_indices = random.sample(range(len(X_test)), 10)
        for i in random_indices:
            sample = X_test[i].tolist()
            result, time_taken = send_request(sample, classifier)

            predictions.append(result['prediction'])
            all_times.append(time_taken)

        average_time = np.mean(all_times)
        std_time = np.std(all_times)

        self.result_label_prediction.config(text=f"Rezultat (1: Trafic normal, -1: Trafic suspicios, 0: Testare fără algoritm): {result['prediction']}")
        self.result_label_time.config(text=f"Timp de detecție: {average_time:.6f} secunde ± {std_time:.6f} secunde")
        self.result_label_avg_operations_per_second.config(text=f"Număr mediu de operații pe secundă: {1 / average_time:.2f}")


# Bucla principala de rulare a aplicatiei cu interfata

if __name__ == "__main__":
    root = tk.Tk()
    app = IDSApp(root)
    root.mainloop()
