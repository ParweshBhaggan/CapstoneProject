import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


class DataVisualizer:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

    
    #Clinical Data Visuals
    # ----------------------------
    def plot_class_distribution(self):
        plt.figure()
        self.df['subtype'].value_counts().plot(kind='bar', color='purple')
        plt.title("Subtype Distribution")
        plt.xlabel("Subtype")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid()
        plt.show()

    def plot_feature_distributions(self):
        for feature in ['gender', 'ESR1', 'PGR', 'ERBB2']:
            plt.figure()
            self.df[feature].value_counts().plot(kind='bar', color='skyblue')
            plt.title(f"{feature} Distribution")
            plt.xlabel(feature)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.grid()
            plt.show()

    def plot_age_distribution(self):
        plt.figure()
        plt.hist(self.df['age'], bins=20, color='orange', edgecolor='black')
        plt.title("Age Distribution")
        plt.xlabel("Age")
        plt.ylabel("Number of Patients")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_feature_vs_age(self, feature):
        if feature not in self.df.columns:
            print(f"Feature '{feature}' not found.")
            return
        plt.figure()
        sns.stripplot(data=self.df, x=feature, y='age', alpha=0.6)
        plt.title(f"Age vs {feature}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_grouped_scatter(self, x, y, group_by='subtype'):
        if x not in self.df.columns or y not in self.df.columns or group_by not in self.df.columns:
            print("❌ Invalid column(s).")
            return
        plt.figure()
        sns.scatterplot(data=self.df, x=x, y=y, hue=group_by)
        plt.title(f"{y} vs {x} by {group_by}")
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    
    # Model Evaluation Visuals
   
    def plot_model_scores(self, scores_dict):
        if not scores_dict:
            print("No scores provided.")
            return
        names = list(scores_dict.keys())
        scores = list(scores_dict.values())
        plt.figure()
        sns.barplot(x=scores, y=names, palette='viridis')
        plt.xlabel("F1 Score (weighted)")
        plt.title("Model Performance Comparison")
        plt.xlim(0, 1.05)
        plt.grid(True, axis='x')
        plt.tight_layout()
        plt.show()

    def plot_feature_importances(self, model, feature_names):
        if not hasattr(model, "feature_importances_"):
            print("Model does not support feature importances.")
            return
        importances = model.feature_importances_
        sorted_idx = importances.argsort()[::-1]
        plt.figure()
        sns.barplot(x=importances[sorted_idx], y=[feature_names[i] for i in sorted_idx])
        plt.title("Feature Importances")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.grid()
        plt.show()

    def plot_confusion_matrix(self, model, X, y, class_names):
        plt.figure()
        disp = ConfusionMatrixDisplay.from_estimator(
            model, X, y, display_labels=class_names,
            cmap='Blues', values_format='d'
        )
        plt.title("Confusion Matrix")
        plt.grid(False)
        plt.tight_layout()
        plt.show()
