from data_services import DataCreator
from ml_services import MachineLearningService
from data_visualization import DataVisualizer
from menu_controller import MenuController  # <- new script

# Paths
root = "datasets/"
data_json = root + "data.json"
csv_path = root + "filtered_data.csv"

#Create and export data
data_creator = DataCreator(data_json)
data_creator.export_data_to_csv(csv_path)

#Train models
ml_service = MachineLearningService(csv_path)
ml_service.run()

#Launch Menu
visualizer = DataVisualizer(csv_path)
menu = MenuController(ml_service, visualizer)
menu.main_menu()
