from data_services import DataCreator

root_file_path = "datasets/"
data_file_path = root_file_path + "data.json"
data_export_file_path = root_file_path + "filtered_data.csv"

data_creator = DataCreator(data_file_path)

#export data
data_creator.export_data_to_csv(data_export_file_path)


