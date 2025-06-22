import json,csv
from data_models import *

"""
Data Services for handling the raw JSON dataset.

This module defines services to process raw clinical JSON data by mapping it into 
structured models, filtering relevant attributes, and generating a cleaner dataset 
ready for preprocessing and machine learning model training.

Components:
- DataMapper: Maps raw JSON fields into Python objects.
- DataFilter: Filters and extracts clinical features such as molecular test results.
- DataClassification: Classifies the data to subtypes.
- DataCreator: Generates the data by loading raw data, processing it, and exporting it 
  into CSV format for training the model.

"""

class DataMapper:
    """
    Create objects from raw JSON data to finally build a Patient object out of the data and it's relevant objects.
    """

    def map_diagnosis_data(self, data):
        """
        Create Diagnosis object from raw JSON data.
        """
        diagnoses_data = data.get("diagnoses", [])
        diagnosis_list = []
        treatments_list = []

        for diagnosis in diagnoses_data:
            tissue_or_organ_of_origin = diagnosis.get("tissue_or_organ_of_origin", "Unknown")
            primary_diagnosis = diagnosis.get("primary_diagnosis", "Unknown")
            state = diagnosis.get("state", "Unknown")
            method_of_diagnosis = diagnosis.get("method_of_diagnosis", "Unknown")
            submitter_id = diagnosis.get("submitter_id", "Unknown")
            classification_of_tumor = diagnosis.get("classification_of_tumor", "Unknown")
            treatment_list = self.map_treatment_data(diagnosis)

            diagnosis_obj = Diagnosis(tissue_or_organ_of_origin, primary_diagnosis, state, method_of_diagnosis, submitter_id, classification_of_tumor  )
            diagnosis_list.append(diagnosis_obj)
            treatments_list.append(treatment_list)
        
        return diagnosis_list, treatments_list
        

    def map_demographic_data(self, data):
        """
        Create Demographic object from raw JSON data.
        """
        demographic_data = data.get("demographic", {})
        gender = demographic_data.get("gender", "Unknown")
        age = demographic_data.get("age_at_index", "Unknown")
        race = demographic_data.get("race", "Unknown")
        vital_status = demographic_data.get("vital_status", "Unknown")
        demographic_obj = Demographic(gender, age, race, vital_status)

        return demographic_obj
    
    def map_treatment_data(self, data):
        """
        Create Treatment object from raw JSON data.
        """
        treatments_data = data.get("treatments", [])
        treatment_list = []

        for treatment in treatments_data:
            treatment_intent_type = treatment.get("treatment_intent_type", "Unknown")
            treatment_type = treatment.get("treatment_type", "Unknown")
            state = treatment.get("state", "Unknown")
            treatment_or_therapy = treatment.get("treatment_or_therapy", "Unknown")

            treatment_obj = Treatment(treatment_intent_type, treatment_type, state, treatment_or_therapy)
            treatment_list.append(treatment_obj)
        
        return treatment_list
        

    def map_molecular_data(self, data):
        """
        Create Molecular object from raw JSON data.
        """
        follow_ups = data.get("follow_ups", [])
        molecular_data_list = []
        
        for follow_up in follow_ups:
            molecular_data = follow_up.get("molecular_tests", [])
            for molecular in molecular_data:
                analysis_method = molecular.get("molecular_analysis_method", "Unknown")
                test_result = molecular.get("test_result", "Unknown")
                gene_symbol = molecular.get("gene_symbol", "Unknown")

                molecular_obj = Molecular(analysis_method, test_result, gene_symbol)
                molecular_data_list.append(molecular_obj)
        
        return molecular_data_list

    def map_patient_data(self, data):
        """
        Build Patient object from raw JSON data and it's relevant objects.
        """
        disease_type = data.get("disease_type", "Unknown")
        project_id = data.get("project", {}).get("project_id", "Unknown")
        submitter_id = data.get("submitter_id", "Unknown")
        consent_type = data.get("consent_type", "Unknown")
        
        diagnoses_output = self.map_diagnosis_data(data)
        diagnoses = diagnoses_output[0]
        treatments = diagnoses_output[1]

        demographic = self.map_demographic_data(data)
        molecular = self.map_molecular_data(data)

        patient = Patient(disease_type, project_id, submitter_id, consent_type)
        patient.diagnosis = diagnoses
        patient.treatment = treatments
        patient.demographic = demographic
        patient.molecular = molecular

        return patient
    
class DataFilter:
    
    """
    Filters data from the gene symbols and test results.
    """

    def get_molecular_gene_result_filtered(self, list_patients):
        gene_symbols = ['PGR', 'ESR1', 'ERBB2']
        test_results = {'Negative', 'Positive'}
        
        
        patients = []
        for patient in list_patients:
            gene_result_map = {}
            for i, molecular in enumerate(patient.molecular):
                if molecular.gene_symbol in gene_symbols and molecular.test_result in test_results:
                    gene_result_map[molecular.gene_symbol] = molecular.test_result

            if not gene_result_map:
                continue

            final_results = [
            {'Gene': gene, 'Result': result}
            for gene, result in gene_result_map.items()
            ]

            patient_data = {'Patient' : patient.submitter_id, 'Result': final_results}
            patients.append(patient_data)
        return patients
    
    
            
class DataClassification:
    
    """
    Classifies the data to subtypes.
    """
    def subtypes_classification(self, filtered_data):
        subtype_patients = []

        for entry in filtered_data:
            submitter_id = entry['Patient']
            gene_map = {res['Gene']: res['Result'] for res in entry['Result']}

            if not all(marker in gene_map for marker in ['ESR1', 'PGR', 'ERBB2']):
                continue

            esr1 = gene_map['ESR1']
            pgr = gene_map['PGR']
            erbb2 = gene_map['ERBB2']

            if esr1 == 'Positive' and pgr == 'Positive' and erbb2 == 'Negative':
                subtype = 'Luminal A'
            elif (esr1 == 'Positive' or pgr == 'Positive') and erbb2 == 'Positive':
                subtype = 'Luminal B'
            elif esr1 == 'Negative' and pgr == 'Negative' and erbb2 == 'Positive':
                subtype = 'HER2-enriched'
            elif esr1 == 'Negative' and pgr == 'Negative' and erbb2 == 'Negative':
                subtype = 'Triple Negative'
            else:
                continue

            subtype_patients.append({'Patient': submitter_id, 'Subtype': subtype})

        return subtype_patients 
        
    

class DataCreator:
    """
    Generates the data and export to CSV format.
    """
    def __init__(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

        self.data = data
        self.data_mapper = DataMapper()
        self.data_filter = DataFilter()
        self.data_classification = DataClassification()

    def export_data_to_csv(self, filepath):
        
        list_patients = self.get_all_patients_data()
        #fetch filtered data
        filtered_data = self.data_filter.get_molecular_gene_result_filtered(list_patients)
        #classify and fetch filtered data
        subtype_data = self.data_classification.subtypes_classification(filtered_data)
        
        subtype_map = {entry['Patient']: entry['Subtype'] for entry in subtype_data}

        rows = []
        for patient in list_patients:
            
            #skip patients that can't be classified
            if patient.submitter_id not in subtype_map:
                continue
            
            #Only accept valid gene results
            gene_results = {gene: "Unknown" for gene in ['ESR1', 'PGR', 'ERBB2']}
            for molecular in patient.molecular:
                if molecular.gene_symbol in gene_results and molecular.test_result in {'Positive', 'Negative'}:
                    gene_results[molecular.gene_symbol] = molecular.test_result

            #Skip if any gene is missing
            if any(val == "Unknown" for val in gene_results.values()):
                continue

            demographic = patient.demographic
            row = {
                "patient_id": patient.submitter_id,
                "gender": str(demographic.gender).upper(),
                "age": str(demographic.age).upper(),
                "ESR1": str(gene_results["ESR1"]).upper(),
                "PGR": str(gene_results["PGR"]).upper(),
                "ERBB2": str(gene_results["ERBB2"]).upper(),
                "subtype": str(subtype_map[patient.submitter_id]).upper()
            }

            rows.append(row)

        # CSV Data columns
        fieldnames = ["patient_id", "gender", "age", "ESR1", "PGR", "ERBB2", "subtype"]

        # Export CSV
        with open(filepath, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Exported {len(rows)} records to: {filepath}")
    
    def get_all_patients_data(self):
        list_patients = []

        for data in self.data:
            patient = self.data_mapper.map_patient_data(data)
            list_patients.append(patient)
        return list_patients  

   
    
    