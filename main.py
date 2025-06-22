from data_services import DataCreator
from ml_services import MachineLearningService

root_file_path = "datasets/"
data_file_path = root_file_path + "data.json"
data_export_file_path = root_file_path + "filtered_data.csv"

#Export data
print("Exporting data...")
data_creator = DataCreator(data_file_path)
data_creator.export_data_to_csv(data_export_file_path)

#Train model
ml_service = MachineLearningService(data_export_file_path)
ml_service.run()


def external_input_prompt():
    def select_option(prompt, options):
        print(f"\n{prompt}")
        for key, val in options.items():
            print(f"{key}. {val}")
        print("0. Back to main menu")

        choice = input("Enter your choice: ").strip()
        if choice == '0':
            return None
        return options.get(choice)

    # Gender selection
    gender = select_option("Select Gender", {'1': 'FEMALE', '2': 'MALE'})
    if gender is None:
        return

    # Age validation
    while True:
        try:
            age = int(input("Enter Age (1-100): ").strip())
            if 1 <= age <= 100:
                break
            else:
                print("Age must be between 1 and 100.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # ESR1
    esr1 = select_option("Select ESR1 Status", {'1': 'POSITIVE', '2': 'NEGATIVE'})
    if esr1 is None:
        return

    # PGR
    pgr = select_option("Select PGR Status", {'1': 'POSITIVE', '2': 'NEGATIVE'})
    if pgr is None:
        return

    # ERBB2
    erbb2 = select_option("Select ERBB2 Status", {'1': 'POSITIVE', '2': 'NEGATIVE'})
    if erbb2 is None:
        return

    input_dict = {
        'gender': gender,
        'age': age,
        'ESR1': esr1,
        'PGR': pgr,
        'ERBB2': erbb2
    }

    print("\n--- Prediction using best model ---")
    ml_service.external_test(input_dict, model_name=ml_service.best_model_name)
    input("\nPress ENTER to return to main menu...")



def display_model_stats_menu():
    while True:
        print("\n=== Model Stats Menu ===")
        print("1. Display all models")
        print("2. Display model performance summary")
        print("3. Display hyperparameter tuning results")
        print("4. Display best model")
        print("0. Back to main menu")

        choice = input("Enter your choice: ").strip()

        if choice == '1':
            ml_service.display_available_models()
            input("\nPress ENTER to return...")
        elif choice == '2':
            ml_service.display_model_performance_summary()
            input("\nPress ENTER to return...")
        elif choice == '3':
            ml_service.display_tuning_results()
            input("\nPress ENTER to return...")
        elif choice == '4':
            ml_service.display_best_model()
            input("\nPress ENTER to return...")
        elif choice == '0':
            break
        else:
            print("Invalid choice. Please try again.")


def main_menu():
    while True:
        print("\n=== MAIN MENU ===")
        print("1. Enter new person data (predict with best model)")
        print("2. Show models")
        print("3. Display model stats")
        print("4. Display best model")
        print("0. Exit")

        choice = input("Enter your choice: ").strip()

        if choice == '1':
            external_input_prompt()
        elif choice == '2':
            ml_service.display_available_models()
            input("\nPress ENTER to return to main menu...")
        elif choice == '3':
            display_model_stats_menu()
        elif choice == '4':
            ml_service.display_best_model()
            input("\nPress ENTER to return to main menu...")
        elif choice == '0':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


# Start the menu
main_menu()
