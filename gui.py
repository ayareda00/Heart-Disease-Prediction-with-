import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from tkinter import PhotoImage

# Global variables
data_cleaned = None
X_train, X_test, y_train, y_test = None, None, None, None
models = {}

# Functions
def upload_file():
    global data_cleaned, X_train, X_test, y_train, y_test, models

    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return

    try:
        data = pd.read_csv(file_path)

        # Data Cleaning
        data_cleaned = data.drop_duplicates()
        data_cleaned['sex'] = LabelEncoder().fit_transform(data_cleaned['sex'])

        # Splitting the dataset
        X = data_cleaned.drop('target', axis=1)
        y = data_cleaned['target']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Save the scaler
        joblib.dump(scaler, 'scaler.pkl')

        # Train models
        models['Logistic Regression'] = LogisticRegression().fit(X_train, y_train)
        models['Decision Tree'] = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
        models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
        models['Gradient Boosting'] = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
        models['KNN'] = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
        models['SVM'] = SVC(kernel='linear', random_state=42).fit(X_train, y_train)
        models['Naive Bayes'] = GaussianNB().fit(X_train, y_train)
        models['CatBoost'] = CatBoostClassifier(iterations=100, learning_rate=0.05, depth=6, verbose=0).fit(X_train, y_train)
        models['SGD'] = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3).fit(X_train, y_train)

        messagebox.showinfo("Success", "File uploaded and models trained successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def view_stats():
    if data_cleaned is None:
        messagebox.showerror("Error", "No dataset loaded.")
        return

    stats_window = tk.Toplevel(root)
    stats_window.title("Dataset Statistics")
    stats_window.geometry("600x500")

    stats = {
        'Shape': data_cleaned.shape,
        'Columns': list(data_cleaned.columns),
        'Null Values': data_cleaned.isnull().sum().to_dict(),
        'Duplicates': data_cleaned.duplicated().sum()
    }

    # Create a Frame for better organization
    stats_frame = tk.Frame(stats_window)
    stats_frame.pack(padx=40, pady=20)

    # Set a nice font style
    title_font = ("Arial", 32, "bold")
    content_font = ("Arial", 18)

    # Title Label centered
    tk.Label(stats_frame, text="Statistics", font=title_font, fg="black").grid(row=0, column=0, columnspan=1, pady=20)

    # Display each statistic
    row = 1
    for key, value in stats.items():
        tk.Label(stats_frame, text=key + ":", font=content_font).grid(row=row, column=0, sticky="w", padx=10, pady=5)

        # Handle values with more details (like 'Null Values' which are dictionaries)
        if isinstance(value, dict):
            value_str = "\n".join([f"{col}: {val}" for col, val in value.items()])
            text_widget = tk.Text(stats_frame, height=10, width=50, font=content_font, wrap=tk.WORD)
            text_widget.insert(tk.END, value_str)
            text_widget.config(state=tk.DISABLED)
            text_widget.grid(row=row, column=1, padx=10, pady=5)
        else:
            tk.Label(stats_frame, text=str(value), font=content_font).grid(row=row, column=1, padx=10, pady=5)

        row += 1

    # Add a close button with updated color
    tk.Button(stats_window, text="Close", command=stats_window.destroy, font=("Arial", 22, "bold"),
              bg="#A3C1AD").pack(pady=30)

def train_model():
    if data_cleaned is None:
        messagebox.showerror("Error", "No dataset loaded.")
        return

    def evaluate_model():
        model_name = model_var.get()
        if model_name not in models:
            messagebox.showerror("Error", "Model not found.")
            return

        model = models[model_name]
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        result_window = tk.Toplevel(root)
        result_window.title(f"{model_name} Results")

        tk.Label(result_window, text=f"Accuracy: {accuracy * 100:.2f}%", font=("Arial", 14, "bold")).pack(pady=10)
        tk.Label(result_window, text="Classification Report:", font=("Arial", 14, "bold")).pack()
        text_widget = tk.Text(result_window, height=15, width=80)
        text_widget.insert(tk.END, report)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack()
        tk.Button(result_window, text="Close", command=result_window.destroy, bg="#A3C1AD", font=("Arial", 12, "bold")).pack(pady=10)

    model_window = tk.Toplevel(root)
    model_window.title("Select Model")

    # Create a Frame for the Model Selection to fill the window
    frame = tk.Frame(model_window)
    frame.pack(fill="both", expand=True, padx=20, pady=20)

    tk.Label(frame, text="Choose a model to evaluate:", font=("Arial", 18, "bold"), pady=20).pack()

    # Variable for model selection
    model_var = tk.StringVar(value=list(models.keys())[0])

    # Make the Radiobuttons bigger and spread them across the page
    for model_name in models.keys():
        tk.Radiobutton(frame, text=model_name, variable=model_var, value=model_name,
                       font=("Arial", 16), width=20, anchor="w", padx=20).pack(fill="x", pady=10)

    # Evaluate Button with larger size
    tk.Button(frame, text="Evaluate", command=evaluate_model, bg="#A3C1AD", font=("Arial", 16, "bold"),
              width=20, height=2).pack(pady=20)

def predict_label():
    if not models:
        messagebox.showerror("Error", "No trained models found. Please upload a dataset and train models first.")
        return

    def make_prediction():
        try:
            # Collect user input
            input_values = [float(entry.get()) for entry in input_entries]

            # Load the scaler to preprocess the data
            scaler = joblib.load('scaler.pkl')
            input_scaled = scaler.transform([input_values])

            # Use the selected model for prediction
            selected_model = model_var.get()
            if selected_model not in models:
                messagebox.showerror("Error", "Invalid model selected.")
                return

            model = models[selected_model]
            prediction = model.predict(input_scaled)
            result = "Predicted Label: " + str(prediction[0])

            # Display the result
            result_label.config(text=result)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    # Create a new window for input and prediction
    predict_window = tk.Toplevel(root)  # Ensure root is defined in the main application
    predict_window.title("Predict Label")

    # Example Input Fields (All 15 features including Max Heart Rate Reserve and Heart Disease Risk Score)
    tk.Label(predict_window, text="Enter feature values:", font=("Arial", 24)).pack(pady=10)
    input_entries = []
    example_values = {
        'Age': '45',
        'Sex (1=Male, 0=Female)': '1',
        'Chest Pain Type (1, 2, 3, 4)': '3',
        'Resting Blood Pressure': '130',
        'Serum Cholesterol': '200',
        'Fasting Blood Sugar (1 if > 120 mg/dl, 0 otherwise)': '0',
        'Resting Electrocardiographic Results (0, 1, 2)': '1',
        'Maximum Heart Rate Achieved': '150',
        'Exercise Induced Angina (1=Yes, 0=No)': '0',
        'Oldpeak (Depression induced by exercise relative to rest)': '1.5',
        'Slope of the Peak Exercise ST Segment (1, 2, 3)': '2',
        'Number of Major Vessels (0-3)': '1',
        'Thalassemia (3=Normal, 6=Fixed defect, 7=Reversible defect)': '3',
        'Max Heart Rate Reserve': '120',
        'Heart Disease Risk Score (e.g., 0 to 10)': '5'
    }

    for column, example_value in example_values.items():
        frame = tk.Frame(predict_window)
        frame.pack(pady=5)
        tk.Label(frame, text=column, width=40, anchor="w").pack(side=tk.LEFT)
        entry = tk.Entry(frame)
        entry.insert(tk.END, example_value)
        entry.pack(side=tk.LEFT, padx=5)
        input_entries.append(entry)

    # Select model
    tk.Label(predict_window, text="Select a model:", font=("Arial", 24)).pack(pady=20)
    model_var = tk.StringVar(value=list(models.keys())[0])

    model_frame = tk.Frame(predict_window)
    model_frame.pack(pady=10)
    model_list = list(models.keys())
    row = 0
    col = 0

    for i, model_name in enumerate(model_list):
        tk.Radiobutton(model_frame, text=model_name, variable=model_var, value=model_name, font=("Arial", 18)).grid(
            row=row, column=col, padx=20, pady=10, sticky="w")
        if col < 2:  # Three columns per row
            col += 1
        else:
            col = 0
            row += 1

    # Prediction button
    tk.Button(predict_window, text="Predict", command=make_prediction, bg="#A3C1AD", font=("Arial", 20)).pack(pady=10)

    # Result label
    result_label = tk.Label(predict_window, text="", font=("Arial", 20), fg="blue")
    result_label.pack(pady=10)


# Main Application
root = tk.Tk()
root.title("Machine Learning GUI")
root.geometry("800x600")  # Set a default size
root.resizable(True, True)  # Allow resizing

## Load the new background image
background_image = PhotoImage(file="C:\\Users\\ayare\\OneDrive\\Desktop\\WhatsApp_Image_2024-12-22_at_01.33.52_0482652f-removebg-preview.png")

# Create a label to display the image as the background
background_label = tk.Label(root, image=background_image)
background_label.place(relx=1.0, rely=1.0, anchor="se")  # Place the image in the bottom-right corner

# UI Elements (Place label at the bottom center)
tk.Label(root, text="Heart Disease Prediction", font=("Arial", 32, "bold"), pady=20).place(relx=0.5, rely=0.17, anchor="center")

buttons_frame = tk.Frame(root, bg="#F1F1F1", bd=0)  # Set the frame background to match
buttons_frame.pack(expand=True)

button_style = {"bg": "#A3C1AD", "font": ("Arial", 12, "bold"), "width": 25, "pady": 10}

tk.Button(buttons_frame, text="Upload Dataset", command=upload_file, **button_style).pack(pady=10)
tk.Button(buttons_frame, text="View Dataset Statistics", command=view_stats, **button_style).pack(pady=10)
tk.Button(buttons_frame, text="Train and Evaluate Models", command=train_model, **button_style).pack(pady=10)
tk.Button(buttons_frame, text="Predict Label", command=predict_label, **button_style).pack(pady=10)
tk.Button(buttons_frame, text="Exit", command=root.quit, **button_style).pack(pady=10)

root.mainloop()
