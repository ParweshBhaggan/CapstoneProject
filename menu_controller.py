class MenuController:
    def __init__(self, ml_service, visualizer):
        self.ml_service = ml_service
        self.visualizer = visualizer

    def external_input_prompt(self):
        def select_option(prompt, options):
            print(f"\n{prompt}")
            for key, val in options.items():
                print(f"{key}. {val}")
            print("0. Back to main menu")
            choice = input("Enter your choice: ").strip()
            if choice == '0':
                return None
            return options.get(choice)

        gender = select_option("Select Gender", {'1': 'FEMALE', '2': 'MALE'})
        if gender is None: return

        while True:
            try:
                age = int(input("Enter Age (1-100): ").strip())
                if 1 <= age <= 100: break
                else: print("Age must be between 1 and 100.")
            except ValueError:
                print("Invalid number.")

        esr1 = select_option("Select ESR1", {'1': 'POSITIVE', '2': 'NEGATIVE'})
        if esr1 is None: return

        pgr = select_option("Select PGR", {'1': 'POSITIVE', '2': 'NEGATIVE'})
        if pgr is None: return

        erbb2 = select_option("Select ERBB2", {'1': 'POSITIVE', '2': 'NEGATIVE'})
        if erbb2 is None: return

        input_dict = {
            'gender': gender,
            'age': age,
            'ESR1': esr1,
            'PGR': pgr,
            'ERBB2': erbb2
        }

        print("\n--- Prediction using best model ---")
        self.ml_service.external_test(input_dict, model_name=self.ml_service.best_model_name)
        input("\nPress ENTER to return to main menu...")

    def display_model_stats_menu(self):
        while True:
            print("\n=== Model Stats Menu ===")
            print("1. Display all models")
            print("2. Display model performance summary")
            print("3. Display hyperparameter tuning results")
            print("4. Display best model")
            print("0. Back to main menu")

            choice = input("Enter your choice: ").strip()
            if choice == '1':
                self.ml_service.display_available_models()
            elif choice == '2':
                self.ml_service.display_model_performance_summary()
            elif choice == '3':
                self.ml_service.display_tuning_results()
            elif choice == '4':
                self.ml_service.display_best_model()
            elif choice == '0':
                break
            else:
                print("Invalid choice.")
            input("\nPress ENTER to return...")

    def data_visualization_menu(self):
        while True:
            print("\n=== Data Visualization Menu ===")
            print("1. Plot Subtype Distribution")
            print("2. Plot Age Distribution")
            print("3. Plot Feature Distributions (gender, ESR1, etc)")
            print("4. Plot Age vs Selected Feature")
            print("5. Scatter Plot Grouped by Subtype")
            print("6. Plot Model F1 Scores")
            print("7. Plot Feature Importances (Random Forest)")
            print("8. Plot Confusion Matrix (Best Model)")
            print("9. PCA Projection of Clinical Features")  # 👈 NEW
            print("0. Back to main menu")

            choice = input("Enter your choice: ").strip()

            if choice == '1':
                self.visualizer.plot_class_distribution()
            elif choice == '2':
                self.visualizer.plot_age_distribution()
            elif choice == '3':
                self.visualizer.plot_feature_distributions()
            elif choice == '4':
                print("\nSelect Feature to Plot Against Age:")
                options = {'1': 'gender', '2': 'ESR1', '3': 'PGR', '4': 'ERBB2'}
                for k, v in options.items():
                    print(f"{k}. {v}")
                print("0. Back")
                selection = input("Enter your choice: ").strip()
                feature = options.get(selection)
                if feature:
                    self.visualizer.plot_feature_vs_age(feature)
                elif selection != '0':
                    print("Invalid choice.")
            elif choice == '5':
                print("\nSelect X-axis feature:")
                x_options = {'1': 'age', '2': 'gender', '3': 'ESR1', '4': 'PGR', '5': 'ERBB2'}
                for k, v in x_options.items():
                    print(f"{k}. {v}")
                print("0. Back")
                x_sel = input("X-axis choice: ").strip()
                x = x_options.get(x_sel)

                if x is None and x_sel != '0':
                    print("Invalid X-axis choice.")
                    continue
                elif x_sel == '0':
                    continue

                print("\nSelect Y-axis feature:")
                y_options = {'1': 'age', '2': 'gender', '3': 'ESR1', '4': 'PGR', '5': 'ERBB2'}
                for k, v in y_options.items():
                    print(f"{k}. {v}")
                print("0. Back")
                y_sel = input("Y-axis choice: ").strip()
                y = y_options.get(y_sel)

                if y is None and y_sel != '0':
                    print("Invalid Y-axis choice.")
                    continue
                elif y_sel == '0':
                    continue

                self.visualizer.plot_grouped_scatter(x, y, group_by='subtype')
            elif choice == '6':
                self.visualizer.plot_model_scores(
                    {k: v['score'] for k, v in self.ml_service.tuned_model_scores.items()}
                )
            elif choice == '7':
                model = self.ml_service.models.get("Random Forest (Tuned)")
                if model:
                    self.visualizer.plot_feature_importances(model, self.ml_service.X.columns)
                else:
                    print("Random Forest (Tuned) not available.")
            elif choice == '8':
                self.visualizer.plot_confusion_matrix(
                    self.ml_service.best_model,
                    self.ml_service.X,
                    self.ml_service.y,
                    self.ml_service.target.classes_
                )
            elif choice == '9':  # 👈 NEW
                self.visualizer.plot_pca_projection()
            elif choice == '0':
                break
            else:
                print("Invalid option.")
            input("\nPress ENTER to return...")


    def main_menu(self):
        while True:
            print("\n=== MAIN MENU ===")
            print("1. Enter new person data (predict)")
            print("2. Show models")
            print("3. Display model stats")
            print("4. Display best model")
            print("5. Data Visualizations")
            print("0. Exit")

            choice = input("Enter your choice: ").strip()

            if choice == '1':
                self.external_input_prompt()
            elif choice == '2':
                self.ml_service.display_available_models()
                input("\nPress ENTER to return...")
            elif choice == '3':
                self.display_model_stats_menu()
            elif choice == '4':
                self.ml_service.display_best_model()
                input("\nPress ENTER to return...")
            elif choice == '5':
                self.data_visualization_menu()
            elif choice == '0':
                print("Goodbye!")
                break
            else:
                print("Invalid choice.")
