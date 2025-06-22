"""
Clinical Data Classes for Data Preparation

This module defines the data structure classes found from the clinical data. 
These classes are used for structuring, cleaning and preparing the data. Each class encapsulates relevant 
Each class has attributes that are relevant for the data preparation and later to use for the model training.

Classes:
- Patient: Represents a patients data.
- Diagnosis: Represents diagnostic information for a patient.
- Treatment: Represents treatment information for a patient.
- Demographic: Represents demographic information of a patient.
- Molecular: Represents molecular test results for a patient.

"""

class Patient:
    def __init__(self, disease_type, project_id, submitter_id,consent_type):
        
        self.project_id = project_id
        self.submitter_id = submitter_id
        self.disease_type = disease_type
        self.consent_type = consent_type

        self.diagnosis = None
        self.treatment = None
        self.demographic = None
        self.molecular = None
    

class Diagnosis:
  
    def __init__(self, origin, primary_diagnosis, state, method_diagnosis, submitter_id, tumor_classification):
        
        self.tissue_or_organ_of_origin = origin
        self.primary_diagnosis = primary_diagnosis
        self.state = state
        self.method_of_diagnosis = method_diagnosis
        self.submitter_id = submitter_id
        self.classification_of_tumor = tumor_classification
    
class Treatment:
  
    def __init__(self, treatment_intent_type, treatment_type, state, treatment_or_therapy):
        self.treatment_intent_type = treatment_intent_type
        self.treatment_type = treatment_type
        self.state = state
        self.treatment_or_therapy = treatment_or_therapy

class Demographic:
   
    def __init__(self, gender, age, race, vital_status):
        self.gender = gender
        self.age = age
        self.race = race
        self.vital_status = vital_status


class Molecular:
    
    def __init__(self, analysis_method, test_result, gene_symbol):
        self.analysis_method = analysis_method
        self.test_result = test_result
        self.gene_symbol = gene_symbol