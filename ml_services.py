"""
Machine Learning Services for Clinical Data Classification

This module provides all the necessary services required for preparing and training 
a machine learning model using structured clinical data. It encapsulates the core 
ML pipeline steps from preprocessing to final predictions.

"""
#pandas
import pandas as pd

#preprocessing
from sklearn.preprocessing import LabelEncoder

#alogrithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#training
from sklearn.model_selection import train_test_split

#reports and scores
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

#hyper tuning
from sklearn.model_selection import GridSearchCV

#plotting graphs
import matplotlib.pyplot as plt


class MachineLearningService:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

        # Preprocessing
        self.label_encoders = {}
        self.X = None
        self.y = None
        self.target = None

        # Models and scores
        self.models = None
        self.model_names = {
            "Logistic Regression", "SVC", "Random Forest", "K-Nearest Neighbors"
        }
        self.tuned_model_scores = {}

        # State
        self.best_model = None
        self.best_model_name = None
        self.is_trained = False

    def run(self):
        self.model_training()
        self.hyperparameter_tuning(self.models)
        self.final_prediction()
        return self.is_trained

    def preprocess_data(self):
        print("Preprocessing data...")
        self.df = self.df.drop(columns=['patient_id'])
        categorical_cols = ['gender', 'ESR1', 'PGR', 'ERBB2']

        for col in categorical_cols:
            label_encoder = LabelEncoder()
            self.df[col] = label_encoder.fit_transform(self.df[col])
            self.label_encoders[col] = label_encoder

        self.target = LabelEncoder()
        self.df['subtype_encoded'] = self.target.fit_transform(self.df['subtype'])

        self.X = self.df.drop(columns=['subtype', 'subtype_encoded'])
        self.y = self.df['subtype_encoded']
        return self.X, self.y, self.target

    def model_training(self):
        print("Training model started...")
        self.X, self.y, self.target = self.preprocess_data()

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "SVC": SVC(kernel='linear'),
            "Random Forest": RandomForestClassifier(random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier()
        }

        for name, model in self.models.items():
            print("Training model: " + name + "...")
            model.fit(X_train, y_train)
            cross_val_score(model, self.X, self.y, cv=5, scoring='f1_weighted')

    def hyperparameter_tuning(self, models):
        print("Hyperparameter tuning started...")
        param_grids = {
            "Logistic Regression": {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            },
            "SVC": {
                'C': [0.01, 0.1, 1, 10],
                'kernel': ['linear']
            },
            "Random Forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "K-Nearest Neighbors": {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            }
        }

        tuned_models = {}

        for name, model in models.items():
            print("Tuning model: " + name + "...")
            if name not in param_grids:
                continue

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grids[name],
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=0
            )

            grid_search.fit(self.X, self.y)

            tuned_name = name + " (Tuned)"
            tuned_models[tuned_name] = grid_search.best_estimator_
            self.tuned_model_scores[tuned_name] = {
                "params": grid_search.best_params_,
                "score": grid_search.best_score_
            }

        self.models.update(tuned_models)
        self.is_trained = True

    def final_prediction(self):
        scores = {}
        for name, model in self.models.items():
            y_pred = model.predict(self.X)
            f1 = f1_score(self.y, y_pred, average='weighted')
            scores[name] = f1

        self.best_model_name = max(scores, key=scores.get)
        self.best_model = self.models[self.best_model_name]
        print("Model training COMPLETED!")

        

    def display_model_performance_summary(self):
        if not self.models:
            print("No models have been trained yet.")
            return

        print("\n=== Model Performance Summary ===")
        for name, model in self.models.items():
            y_pred = model.predict(self.X)
            print(classification_report(self.y, y_pred, target_names=self.target.classes_))
            f1 = f1_score(self.y, y_pred, average='weighted')
            print(f"{name}: F1 (weighted) = {f1:.4f}")

    def display_tuning_results(self):
        if not self.tuned_model_scores:
            print("No tuning results available.")
            return

        print("\n=== Hyperparameter Tuning Results ===")
        for name, info in self.tuned_model_scores.items():
            print(f"Tuning hyperparameters for: {name}")
            print(f"Best parameters: {info['params']}")
            print(f"Best F1 (weighted): {info['score']:.4f}")

    def display_available_models(self):
        print("\nAvailable Models:")
        for name in self.models.keys():
            print(f"- {name}")

    def display_best_model(self):
        if self.best_model_name:
            y_pred = self.best_model.predict(self.X)
            f1 = f1_score(self.y, y_pred, average='weighted')
            print(f"\nBest Model: {self.best_model_name} with F1 (weighted) = {f1:.4f}")
        else:
            print("No best model has been selected yet.")

    def internal_test(self, model_name="Logistic Regression (Tuned)", sample_size=5):
        if model_name not in self.models:
            print(f"Model '{model_name}' not found.")
            return

        model = self.models[model_name]
        sample = self.X.iloc[:sample_size]
        predictions = model.predict(sample)
        decoded = self.target.inverse_transform(predictions)

        print(f"\nInternal Test using {model_name}")
        print("Sample input:\n", sample)
        print("Predicted subtypes:", decoded)

    def external_test(self, input_dict, model_name="Logistic Regression (Tuned)"):
        if model_name not in self.models:
            print(f"Model '{model_name}' not found.")
            return

        model = self.models[model_name]
        encoded_input = {}

        for col, value in input_dict.items():
            if col in self.label_encoders:
                le = self.label_encoders[col]
                encoded_input[col] = le.transform([value])[0]
            else:
                encoded_input[col] = value

        df_input = pd.DataFrame([encoded_input])
        prediction = model.predict(df_input)
        decoded = self.target.inverse_transform(prediction)

        print("Predicted Subtype:", decoded[0])