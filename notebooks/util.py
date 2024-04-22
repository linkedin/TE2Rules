from sklearn import metrics
import pandas as pd


feature_mapping = {
    'age': ['age'], 
    'capital_gain': ['capital_gain'], 
    'capital_loss': ['capital_loss'], 
    'hours_per_week': ['hours_per_week'], 
    'workclass': ['workclass_Federal_gov', 'workclass_Local_gov', 'workclass_Never_worked', 
                  'workclass_Private', 'workclass_Self_emp_inc', 'workclass_Self_emp_not_inc', 
                  'workclass_State_gov', 'workclass_Without_pay'], 
    'education': ['education_Bachelors', 'education_Doctorate', 'education_HS_grad', 
                  'education_Masters', 'education_Prof_school', 'education_School', 
                  'education_Some_college', 'education_Voc'], 
    'marital_status': ['marital_status_divorced', 'marital_status_married', 'marital_status_not_married'], 
    'occupation': ['occupation_Adm_clerical', 'occupation_Armed_Forces', 'occupation_Craft_repair', 
                   'occupation_Exec_managerial', 'occupation_Farming_fishing', 'occupation_Handlers_cleaners', 
                   'occupation_Machine_op_inspct', 'occupation_Other_service', 'occupation_Priv_house_serv', 
                   'occupation_Prof_specialty', 'occupation_Protective_serv', 'occupation_Sales', 'occupation_Tech_support', 
                   'occupation_Transport_moving'], 
    'relationship': ['relationship_Husband', 'relationship_Not_in_family', 'relationship_Other_relative', 
                     'relationship_Own_child', 'relationship_Unmarried', 'relationship_Wife'], 
    'race': ['race_Amer_Indian_Eskimo', 'race_Asian_Pac_Islander', 'race_Black', 'race_Other', 'race_White'], 
    'sex': ['sex_Female', 'sex_Male'], 
    'native_country': ['native_country_Other', 'native_country_United_States']
}

def get_feature_to_category_mapping():
    feature_to_category_mapping = {}
    for k, v in feature_mapping.items():
        for i in range(len(v)):
            feature_to_category_mapping[v[i]] = k
    return feature_to_category_mapping    

def get_category_mappings_from_data(x, features):
    feature_to_category_map = get_feature_to_category_mapping()
    
    x_feature_map = {} 
    for i in range(len(x)):
        x_feature_map[features[i]] = x[i]

    x_category_map = {}
    for k in x_feature_map.keys():
        category = feature_to_category_map[k]
        x_category_map[category] = []
        for fi in feature_mapping[category]:
            x_category_map[category].append(x_feature_map[fi])
    return x_category_map

def display_input(x, features):
    x_category_map = get_category_mappings_from_data(x, features)
    for k, v in x_category_map.items():
        if(len(v) == 1):
            value = v[0]
            print(k, ":", value)
        else:
            for i in range(len(v)):
                if(v[i] == 1):
                    value = feature_mapping[k][i]
                    value = value[len(k) + 1:]
                    print(k, ":", value)
    return

    