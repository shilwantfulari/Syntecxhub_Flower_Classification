import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# --- CONSOLE STYLING CONSTANTS ---
class Style:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

def load_and_explore_data():
    """Loads Iris dataset, applies visual themes, and shows EDA."""
    print(f"{Style.HEADER}{Style.BOLD}\n========================================")
    print(f" 1. DATA LOADING & EXPLORATION")
    print(f"========================================{Style.ENDC}")
    
    # Set a professional visual theme for the plots
    sns.set_theme(style="whitegrid", palette="husl")
    
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print(f"{Style.BLUE}[INFO] Dataset loaded successfully.{Style.ENDC}")
    print(f"{Style.BLUE}[INFO] Generating Pair Plot... (Check the popup window){Style.ENDC}")
    
    # Generate the Pair Plot with the corrected title position
    g = sns.pairplot(df, hue='species', height=2.5)
    g.fig.suptitle("Iris Dataset Feature Pairs (EDA)", y=1.02, fontsize=16, fontweight='bold')
    plt.show()
    
    return iris.data, iris.target, iris.target_names

def train_and_evaluate(X, y, target_names):
    """Trains models and displays formatted evaluation metrics."""
    print(f"{Style.HEADER}{Style.BOLD}\n========================================")
    print(f" 2. MODEL TRAINING & EVALUATION")
    print(f"========================================{Style.ENDC}")
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Decision Tree": DecisionTreeClassifier()
    }
    
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        trained_models[name] = model
        
        print(f"\nModel: {Style.BOLD}{name}{Style.ENDC}")
        print(f"Accuracy: {Style.GREEN}{acc:.2%}{Style.ENDC}")
        
        # Plot Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Confusion Matrix: {name}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')
        plt.tight_layout() # Ensures everything fits nicely
        plt.show()

    return trained_models

def prediction_cli(models, target_names):
    """Interactive CLI for user predictions."""
    print(f"{Style.HEADER}{Style.BOLD}\n========================================")
    print(f" 3. REAL-TIME PREDICTION (CLI)")
    print(f"========================================{Style.ENDC}")
    print("Type your measurements to predict the flower species.")
    print(f"{Style.WARNING}Format: sepal_len, sepal_wid, petal_len, petal_wid{Style.ENDC}")
    print("(Example: 5.1, 3.5, 1.4, 0.2)")
    
    model = models["Logistic Regression"]
    
    while True:
        user_input = input(f"\n{Style.BOLD}Enter values (or 'q' to exit): {Style.ENDC}")
        
        if user_input.lower() == 'q':
            print(f"\n{Style.GREEN}Goodbye! Project Completed.{Style.ENDC}")
            break
        
        try:
            features = [float(x.strip()) for x in user_input.split(',')]
            
            if len(features) != 4:
                print(f"{Style.FAIL}[ERROR] Please enter exactly 4 numbers separated by commas.{Style.ENDC}")
                continue
                
            prediction = model.predict([features])[0]
            species_name = target_names[prediction]
            
            print(f"--> Predicted Species: {Style.GREEN}{Style.BOLD}{species_name.upper()}{Style.ENDC}")
            
        except ValueError:
            print(f"{Style.FAIL}[ERROR] Invalid input. Please enter numbers only.{Style.ENDC}")

if __name__ == "__main__":
    try:
        X, y, target_names = load_and_explore_data()
        models = train_and_evaluate(X, y, target_names)
        prediction_cli(models, target_names)
    except KeyboardInterrupt:
        print(f"\n\n{Style.FAIL}Program interrupted by user.{Style.ENDC}")