import numpy as np
import pandas as pd

from collections import Counter

from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

def extract_values(df, column, values):
    d = df[df[column].isin(values)]
    print(len(set(d.index)), column)
    return set(d.index), d 

def scan_unique(df, column):
    return df[column].unique()

def col_name(name):
    name = name.lower()
    return data.columns[[name in i.lower() for i in data.columns]]

def model_evaluations(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(classification_report(y_pred, y_test))
    print("F1 Score:", f1_score(y_pred, y_test, average="macro"))
    print("Accuracy:", accuracy_score(y_pred, y_test))
    print(pd.DataFrame(confusion_matrix(y_test, y_pred)))
    
def gcv_rf():
    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 10)]
    max_depth = [int(x) for x in np.linspace(5, 50, num = 10)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]

    grid_params = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}

    model = GridSearchCV(RandomForestClassifier(), grid_params, cv=5, verbose=1, n_jobs=-1)

    return model

def gcv_lr():
    grid_params = {"C": [0.001, 0.01, 0.1, 1, 10, 100], "penalty": ["l1", "l2"]}

    model = GridSearchCV(LogisticRegression(), grid_params, cv=5, verbose=1, n_jobs=-1)

    return model

def gcv_svm():
    grid_params = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1], 'kernel': ['rbf', 'linear']}

    model = GridSearchCV(SVC(), grid_params, cv=5, verbose=1, n_jobs=-1)

    return model

def gcv_mlp():
    grid_params = {
        'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],
    }

    model = GridSearchCV(MLPClassifier(), grid_params, cv=5, verbose=1, n_jobs=-1)

    return model

def drop_missing_columns(dataset, missing_per=0.8):
    print ('Before dataset.shape', dataset.shape)
    k = pd.DataFrame(dataset.notna().sum()/dataset.shape[0])
    k = k[k[0]>missing_per]
    dense_attributes = k.index.tolist()
    dataset = dataset[dense_attributes]
    print ('After dataset.shape', dataset.shape)
    return dataset
    


attributes_considered = ['GENDER', #Gender
                        
                         # Education
                         'PRESENT STATE 13-46: (016) AGE LEFT SCHOOL',
                         'MOCA: YEARS OF EDUCATION',
                         'PRESENT STATE 13-46: (017) YEARS IN FURTHER EDUCATION',
                         
                         # Age
                         'PRESENT STATE 13-46: (014) AGE KNOWN',
                         'Age At Episode',
                         
                         # Stroke
                         'HISTORY PATIENT 74-119: (076) HISTORY OF STROKE',
                         'MEDICAL ASSESSMENT V 2010: STROKE',
                         'HACHINSKI ISCHAEMIC: HISTORY OF STROKE',
                         'OPTIMA DIAGNOSES V 2010: CERBRO-VASCULAR DISEASE PRESENT',
                         
                         # Diabetes
                         'GENERAL INFORMATION: DIABETES: DURATION',
                         'GENERAL INFORMATION: DIABETES',
                         
                         # BMI
                         'CLINICAL BACKGROUND: BODY MASS INDEX',
                         
                         # Depression
                         'CAMDEX ADMINISTRATION 1-12: EST OF SEVERITY OF DEPRESSION',
                         'PRESENT STATE 13-46: (036) FEELING DEPRESSED',
                         'INTERVIEWER OBS 188-212: (196) DEPRESSED MOOD',
                         'DIAGNOSIS 334-351: SEVERITY OF DEPRESSION',
                         'MEDICAL ASSESSMENT V 2010: DEPRESSION',
                         'MEDICAL ASSESSMENT V 2010: DEPRESSION TREATED BY DOCTOR',
                         'HACHINSKI ISCHAEMIC: DEPRESSIVE SYMPTOMATOLOGY',
                         'NPI: DEPRESSION/DYSPHORIA: FREQUENCY',
                         'NPI: DEPRESSION/DYSPHORIA: SEVERITY',
                         'NPI: DEPRESSION/DYSPHORIA: DISTRESS',
                         'SPECT SCAN: DIAGNOSTIC ASSESSMENT',
                         'DIAGNOSIS 334-351: DEPRESSIVE ILLNESS',
                         
                         # Cholestrol
                         'BIOCHEMISTRY: CHOLESTEROL',
                         'BIOCHEMISTRY: HDL CHOLESTEROL',
                         'BIOCHEMISTRY: CHOLESTEROL/HDL RATIO',
                         
                         # Head Injury
                         'HISTORY PATIENT 74-119: (077) HISTORY OF HEAD INJURY',
                         'MEDICAL ASSESSMENT V 2010: HEAD INJURY',
                         
                         
                         # Smoking
                         'SMOKING: SMOKING',
                         'SMOKING: AVERAGE PER WEEK',
                         'SMOKING: PIPES OR CIGARS',
                         'SMOKING: CIGARETTES',
                         'SMOKING: CIGARETTES: NO. PER YEAR',
                         'SMOKING: TWO YEARS AGO',
                         'SMOKING: TEN YEARS AGO', 
                         'SMOKING: TWENTY YEARS AGO',
                         'SMOKING: THIRTY YEARS AGO', 
                         'SMOKING: AGE STARTED CIGARETTES',
                         'SMOKING: TEA PER DAY', 'SMOKING: COFFEE PER DAY',
                         'SMOKING: ALCOHOL IN PAST 12 MONTHS',
                         'SMOKING: AVERAGE PER WEEK', 
                         'SMOKING: COMPARED TO 5 YEARS AGO',
                         'SMOKING: NON-DRINKER ALMOST ALWAYS',
                         'MEDICAL ASSESSMENT V 2010: AGE STOPPED TOBACCO',
                         
                         # Memory Problems
                         'SHORTENED CAMBRIDGE ADL: FORGET TO PASS ON PHONE MESSAGES',
                         'PRESENT STATE 47-73: (058) MEMORY PROBLEM',
                         'CLINICAL DEMENTIA RATING: MEMORY',
                         'PRESENT STATE 47-73: (062) DURATION OF MEMORY PROBLEMS',
                         'PRESENT STATE 47-73: (063) ONSET OF MEMORY PROBLEMS',
                         'PRESENT STATE 47-73: (064) CHANGE IN MEMORY PROBLEMS',
                         'GDS: MEMORY PROBLEMS',
                         'CAMDEX SCORES: CAMCOG REMOTE MEMORY SCORE',
                         'CAMDEX SCORES: CAMCOG RECENT MEMORY SCORE',
                         'CAMDEX SCORES: CAMCOG LEARNING MEMORY SCORE',
                         'MEDICAL ASSESSMENT V 2010: MEMORY',
                         'SHORTENED CAMBRIDGE ADL: DIFFICULTY WITH MEMORY',
                         'SHORTENED CAMBRIDGE ADL: POOR DAY-TO-DAY MEMORY',
                         
                         # Alcohol
                         'SMOKING: ALCOHOL IN PAST 12 MONTHS',
                         'MEDICAL ASSESSMENT V 2010: ALCOHOL CONSUPTION',
                         'MEDICAL ASSESSMENT V 2010: AGE STOPPED ALCOHOL',
                         
                         # Cognitive Activity
                        # 'COGNITIVE EXAM 120-161: COGNITIVE EXAM 120-161',
                         'COGNITIVE EXAM 120-161: (120) IDENTIFIES DAY OF WEEK',
                         'COGNITIVE EXAM 120-161: (121) IDENTIFIES DATE',
                         'COGNITIVE EXAM 120-161: (122) IDENTIFIES MONTH',
                         'COGNITIVE EXAM 120-161: (123) IDENTIFIES YEAR',
                         'COGNITIVE EXAM 120-161: (124) IDENTIFIES SEASON',
                         'COGNITIVE EXAM 120-161: (125) IDENTIFIES COUNTY',
                         'COGNITIVE EXAM 120-161: (126) IDENTIFIES TOWN',
                         'COGNITIVE EXAM 120-161: (127) IDENTIFIES STREETS/COUNTRY',
                         'COGNITIVE EXAM 120-161: (128) IDENTIFIES FLOOR',
                         'COGNITIVE EXAM 120-161: (129) IDENTIFIES PRESENT PLACE',
                         'COGNITIVE EXAM 120-161: (130) COMPREHENDS NOD',
                         'COGNITIVE EXAM 120-161: (131) COMPREHENDS TOUCH',
                         'COGNITIVE EXAM 120-161: (132) COMPREHENDS LOOK',
                         'COGNITIVE EXAM 120-161: (133) COMPREHENDS TAP',
                         'COGNITIVE EXAM 120-161: (134) COMPREHENDS HOTEL',
                         'COGNITIVE EXAM 120-161: (135) COMPREHENDS VILLAGE',
                         'COGNITIVE EXAM 120-161: (136) COMPREHENDS RADIO',
                         'COGNITIVE EXAM 120-161: (137) IDENTIFIES OBJECTS',
                         'COGNITIVE EXAM 120-161: (137) IDENTIFIES OBJECTS: PENCIL',
                         'COGNITIVE EXAM 120-161: (137) IDENTIFIES OBJECTS: WATCH',
                         'COGNITIVE EXAM 120-161: (138) NAMES PICTURES',
                         'COGNITIVE EXAM 120-161: (138) NAMES PICTURES: SHOE',
                         'COGNITIVE EXAM 120-161: (138) NAMES PICTURES: TYPEWRITER',
                         'COGNITIVE EXAM 120-161: (138) NAMES PICTURES: SCALES',
                         'COGNITIVE EXAM 120-161: (138) NAMES PICTURES: SUITCASE',
                         'COGNITIVE EXAM 120-161: (138) NAMES PICTURES: BAROMETER',
                         'COGNITIVE EXAM 120-161: (138) NAMES PICTURES: LAMP',
                         'COGNITIVE EXAM 120-161: (139) NUMBER OF ANIMALS LISTED',
                         'COGNITIVE EXAM 120-161: (139) NUMBER OF ANIMALS LISTED: SCORE',
                         'COGNITIVE EXAM 120-161: (140) DEFINES HAMMER',
                         'COGNITIVE EXAM 120-161: (144) REPETITION',
                         'COGNITIVE EXAM 120-161: (146) RECALLS OBJECTS',
                         'COGNITIVE EXAM 120-161: (146) RECALLS OBJECTS: SHOE',
                         'COGNITIVE EXAM 120-161: (146) RECALLS OBJECTS: TYPEWRITER',
                         'COGNITIVE EXAM 120-161: (146) RECALLS OBJECTS: SCALES',
                         'COGNITIVE EXAM 120-161: (146) RECALLS OBJECTS: SUITCASE',
                         'COGNITIVE EXAM 120-161: (146) RECALLS OBJECTS: BAROMETER',
                         'COGNITIVE EXAM 120-161: (146) RECALLS OBJECTS: LAMP',
                         'COGNITIVE EXAM 120-161: (147) RECOGNISES PICTURES: SHOE',
                         'COGNITIVE EXAM 120-161: (147) RECOGNISES PICTURES: SCALES',
                         'COGNITIVE EXAM 120-161: (147) RECOGNISES PICTURES: BAROMETER',
                         'COGNITIVE EXAM 120-161: (148) REMEMBERS WW1 DATE',
                         'COGNITIVE EXAM 120-161: (149) REMEMBERS WW2 DATE',
                         'COGNITIVE EXAM 120-161: (150) REMEMBERS HITLER',
                         'COGNITIVE EXAM 120-161: (151) REMEMBERS STALIN',
                         'COGNITIVE EXAM 120-161: (152) REMEMBERS MAE WEST',
                         'COGNITIVE EXAM 120-161: (153) REMEMBERS LINDBERGH',
                         'COGNITIVE EXAM 120-161: (154) KNOWS MONARCH',
                         'COGNITIVE EXAM 120-161: (155) KNOWS HEIR TO THRONE',
                         'COGNITIVE EXAM 120-161: (156) KNOWS PRIME MINISTER',
                         'COGNITIVE EXAM 120-161: (157) KNOWS RECENT NEWS ITEM',
                         'COGNITIVE EXAM 120-161: (158) REGISTERS OBJECTS',
                         'COGNITIVE EXAM 120-161: (158) REGISTERS OBJECTS 1: APPLE',
                         'COGNITIVE EXAM 120-161: (158) REGISTERS OBJECTS 3: PENNY',
                         'COGNITIVE EXAM 120-161: (158) REGISTERS OBJECTS: REPEATS',
                         'COGNITIVE EXAM 120-161: (159) COUNTING BACKWARDS',
                         'COGNITIVE EXAM 120-161: (160A) SPELL BACKWARD',
                         'COGNITIVE EXAM 120-161: (161) RECALLS OBJECTS',
                         'COGNITIVE EXAM 120-161: (161) RECALLS OBJECTS 1: APPLE',
                         'COGNITIVE EXAM 120-161: (161) RECALLS OBJECTS 2: TABLE',
                         'COGNITIVE EXAM 120-161: (161) RECALLS OBJECTS 3: PENNY',
                        # 'COGNITIVE EXAM 162-187: COGNITIVE EXAM 162-187',
                         'COGNITIVE EXAM 162-187: (162) READING COMPREHENSION 1',
                         'COGNITIVE EXAM 162-187: (163) READING COMPREHENSION 2',
                         'COGNITIVE EXAM 162-187: (164) DRAWS PENTAGON',
                         'COGNITIVE EXAM 162-187: (165) DRAWS SPIRAL',
                         'COGNITIVE EXAM 162-187: (166) DRAWS HOUSE',
                         'COGNITIVE EXAM 162-187: (167) CLOCK DRAWING',
                         'COGNITIVE EXAM 162-187: (167) CLOCK DRAWING: CIRCLE',
                         'COGNITIVE EXAM 162-187: (167) CLOCK DRAWING: NUMBERS',
                         'COGNITIVE EXAM 162-187: (167) CLOCK DRAWING: TIME',
                         'COGNITIVE EXAM 162-187: (168) WRITES A SENTENCE',
                         'COGNITIVE EXAM 162-187: (169) PRAXIS - PAPER',
                         'COGNITIVE EXAM 162-187: (169) PRAXIS - PAPER: RIGHT HAND',
                         'COGNITIVE EXAM 162-187: (169) PRAXIS - PAPER: FOLDS',
                         'COGNITIVE EXAM 162-187: (169) PRAXIS - PAPER: ON LAP',
                         'COGNITIVE EXAM 162-187: (170) PRAXIS - ENVELOPE',
                         'COGNITIVE EXAM 162-187: (171) DICTATION',
                         'COGNITIVE EXAM 162-187: (172) MIME - WAVE',
                         'COGNITIVE EXAM 162-187: (173) MIME - SCISSORS',
                         'COGNITIVE EXAM 162-187: (174) MIME - BRUSHING TEETH',
                         'COGNITIVE EXAM 162-187: (175) IDENTIFIES COIN',
                         'COGNITIVE EXAM 162-187: (176) ADDS UP MONEY',
                         'COGNITIVE EXAM 162-187: (177) SUBTRACTS MONEY',
                         'COGNITIVE EXAM 162-187: (178) RECALLS ADDRESS',
                         'COGNITIVE EXAM 162-187: (178) RECALLS ADDRESS: JOHN',
                         'COGNITIVE EXAM 162-187: (178) RECALLS ADDRESS: BROWN',
                         'COGNITIVE EXAM 162-187: (178) RECALLS ADDRESS: D42',
                         'COGNITIVE EXAM 162-187: (178) RECALLS ADDRESS: WEST',
                         'COGNITIVE EXAM 162-187: (178) RECALLS ADDRESS: BEDFORD',
                         'COGNITIVE EXAM 162-187: (179) SIMILARITIES - FRUIT',
                         'COGNITIVE EXAM 162-187: (180) SIMILARITIES - CLOTHING',
                         'COGNITIVE EXAM 162-187: (181) SIMILARITIES - FURNITURE',
                         'COGNITIVE EXAM 162-187: (182) SIMILARITIES - LIFE',
                         'COGNITIVE EXAM 162-187: (183) RECOGNISES FAMOUS PEOPLE',
                         'COGNITIVE EXAM 162-187: (184) RECOGNISES OBJECTS',
                         'COGNITIVE EXAM 162-187: (184) RECOGNISES OBJECTS: SPECTACLES',
                         'COGNITIVE EXAM 162-187: (184) RECOGNISES OBJECTS: SHOE',
                         'COGNITIVE EXAM 162-187: (184) RECOGNISES OBJECTS: PURSE',
                         'COGNITIVE EXAM 162-187: (184) RECOGNISES OBJECTS: CUP',
                         'COGNITIVE EXAM 162-187: (184) RECOGNISES OBJECTS: TELEPHONE',
                         'COGNITIVE EXAM 162-187: (184) RECOGNISES OBJECTS: PIPE',
                         'COGNITIVE EXAM 162-187: (185) RECOGNISE PERSON',
                         'COGNITIVE EXAM 162-187: (187) PATIENT',
                         'COGNITIVE EXAM 162-187: HANDED',
                         'OPTIMA DIAGNOSES V 2010: COGNITIVE IMPAIRMENT',
                         
                         # Physical Activity
                         'PRESENT STATE 47-73: (050) PHYSICAL SYMPTOMS',
                         'PRESENT STATE 47-73: (057) PHYSICAL PROBLEMS',
                        # 'PHYSICAL EXAM 213-234: PHYSICAL EXAM 213-234',
                         'PHYSICAL EXAM 213-234: (213) BLOOD PRESSURE',
                         'PHYSICAL EXAM 213-234: (213) BLOOD PRESSURE: SYSTOLIC',
                         'PHYSICAL EXAM 213-234: (213) BLOOD PRESSURE: DIASTOLIC',
                         'PHYSICAL EXAM 213-234: (215) TENDON REFLEXES',
                         'PHYSICAL EXAM 213-234: (216) PLANTAR REFLEXES',
                         'PHYSICAL EXAM 213-234: (217) HEMIPARESIS',
                         'PHYSICAL EXAM 213-234: (218) GAIT',
                         'PHYSICAL EXAM 213-234: (219) MOBILITY',
                         'PHYSICAL EXAM 213-234: (220) DEAFNESS',
                         'PHYSICAL EXAM 213-234: (221) VISUAL DEFECT',
                         'PHYSICAL EXAM 213-234: (222) TREMOR',
                         'PHYSICAL EXAM 213-234: (223) MANUAL DIFFICULTY',
                         'PHYSICAL EXAM 213-234: (224) ABNORMAL EYE MOVEMENTS',
                         'PHYSICAL EXAM 213-234: (225) SHORTNESS OF BREATH',
                         'PHYSICAL EXAM 213-234: (226) FULL BLOOD COUNT',
                         'PHYSICAL EXAM 213-234: (227) B12 OR FOLATE',
                         'PHYSICAL EXAM 213-234: (228) THYROID FUNCTION TESTS',
                         'PHYSICAL EXAM 213-234: (229) UREA AND ELECTROLYTES',
                         'PHYSICAL EXAM 213-234: (230) SKULL XRAY OR SPECT SCAN',
                         'PHYSICAL EXAM 213-234: (231) LIVER FUNCTION TESTS',
                         'PHYSICAL EXAM 213-234: (232) CT OR MRI SCAN',
                         'PHYSICAL EXAM 213-234: (233) VDRL',
                         'PHYSICAL EXAM 213-234: (234) CAUSES OF DEMENTIA EXCLUDED',
                         'PHYSICAL EXAM 213-234: SUBJECT ON MEDICATION',
                         
                         # Social Engagement
                         'GDS: AVOID SOCIAL GATHERINGS',
                         
                         # Systolic BP
                         'PHYSICAL EXAM 213-234: (213) BLOOD PRESSURE: SYSTOLIC',
                         
                         # APOE4 Genetics
                         'PM INFORMATION: APOE', 'PM INFORMATION: APOE: RESULT',
                         
                         # Anxiety
                         'DIAGNOSIS 334-351: ANXIETY/PHOBIC',
                         'PRESENT STATE 47-73: (049) ANXIOUS',
                         'PRESENT STATE 47-73: (051) ANXIOUS SITUATIONS',
                         'INTERVIEWER OBS 188-212: (195) ANXIOUS OR FEARFUL',
                         'MEDICAL ASSESSMENT V 2010: ANXIETY',
                         'NPI: ANXIETY: FREQUENCY',
                         'NPI: ANXIETY: SEVERITY',
                         'NPI: ANXIETY: F X S',
                         'NPI: ANXIETY: DISTRESS',
                         
                         # Aspirin
                         'GENERAL INFORMATION: ASPIRIN: DURATION',
                         'GENERAL INFORMATION: ASPIRIN',
                         
                         # Label
                         'dementia_range']

george_labels_three_years = ['Age At Episode', 'PRESENT STATE 13-46: (016) AGE LEFT SCHOOL',
       'PRESENT STATE 13-46: (018) LIVING ARRANGEMENTS',
       'PRESENT STATE 13-46: (025) WAKES EARLY',
       'PRESENT STATE 13-46: (029) DECISION MAKING DIFFICULT',
       'PRESENT STATE 13-46: (030) PLEASURE LOSS',
       'PRESENT STATE 13-46: (033) CONCENTRATION DIFFICULT',
       'PRESENT STATE 13-46: (034) SLOWED SPEECH',
       'PRESENT STATE 13-46: (035) SLOWED THOUGHT',
       'PRESENT STATE 13-46: (036) FEELING DEPRESSED',
       'PRESENT STATE 13-46: (037) DURATION OF DEPRESSION',
       'PRESENT STATE 13-46: (043) WORTHLESS OR GUILTY',
       'PRESENT STATE 13-46: (044) FUTURE FEELINGS',
       'PRESENT STATE 47-73: (053) DURATION OF PANIC',
       'PRESENT STATE 47-73: (058) MEMORY PROBLEM',
       'PRESENT STATE 47-73: (060) FORGETS NAMES',
       'PRESENT STATE 47-73: (062) DURATION OF MEMORY PROBLEMS',
       'HISTORY PATIENT 74-119: (086) NUMBER OF SONS',
       'HISTORY PATIENT 74-119: (087) NUMBER OF DAUGHTERS',
       'HISTORY PATIENT 74-119: (088) NUMBER OF BROTHERS',
       'HISTORY PATIENT 74-119: (089) NUMBER OF SISTERS',
       'HISTORY PATIENT 74-119: (090) SUBJECTS POSTN. IN FAMILY',
       'HISTORY PATIENT 74-119: (092) MOTHER AGE AT DEATH',
       'HISTORY PATIENT 74-119: (094) FATHER AGE AT DEATH',
       'COGNITIVE EXAM 120-161: (121) IDENTIFIES DATE',
       'COGNITIVE EXAM 120-161: (139) NUMBER OF ANIMALS LISTED',
       'COGNITIVE EXAM 120-161: (139) NUMBER OF ANIMALS LISTED: SCORE',
       'COGNITIVE EXAM 120-161: (146) RECALLS OBJECTS',
       'COGNITIVE EXAM 120-161: (146) RECALLS OBJECTS: SHOE',
       'COGNITIVE EXAM 120-161: (146) RECALLS OBJECTS: TYPEWRITER',
       'COGNITIVE EXAM 120-161: (147) RECOGNISES PICTURES',
       'COGNITIVE EXAM 120-161: (160) SUBTRACTING SEVENS',
       'COGNITIVE EXAM 120-161: (161) RECALLS OBJECTS',
       'COGNITIVE EXAM 162-187: (175) IDENTIFIES COIN',
       'COGNITIVE EXAM 162-187: (178) RECALLS ADDRESS',
       'COGNITIVE EXAM 162-187: (178) RECALLS ADDRESS: D42',
       'COGNITIVE EXAM 162-187: (181) SIMILARITIES - FURNITURE',
       'COGNITIVE EXAM 162-187: (184) RECOGNISES OBJECTS',
       'COGNITIVE EXAM 162-187: (184) RECOGNISES OBJECTS: SHOE',
       'COGNITIVE EXAM 162-187: ACTUAL DURATION OF INTERVIEW',
       #'PHYSICAL EXAM 213-234: PHYSICAL EXAM 213-234',
       'CAMDEX SCORES: MINI MENTAL SCORE', 
       'CAMDEX SCORES: TOTAL CAMCOG SCORE',
       'CAMDEX SCORES: CAMCOG EXPRESSION SCORE',
       'CAMDEX SCORES: CAMCOG REMOTE MEMORY SCORE',
       'CAMDEX SCORES: CAMCOG LEARNING MEMORY SCORE',
       'CAMDEX SCORES: CAMCOG ATTENTION SCORE',
       'OPTIMA DIAGNOSES V 2010: CERBRO-VASCULAR DISEASE PRESENT',
       #'OPTIMA DIAGNOSES V 2010: DIAGNOSTIC CODE', 'BIOCHEMISTRY: CALCIUM',
       'BIOCHEMISTRY: PHOSPHATE', 'BIOCHEMISTRY: ALBUMIN',
       'HAEMATOLOGY: HAEMOGLOBIN', 'HAEMATOLOGY: PLATELETS',
       'HAEMATOLOGY: HAEMATOCRIT', 'HAEMATOLOGY: RED CELL COUNT',
       'HAEMATOLOGY: MCV', 'HAEMATOLOGY: MCH', 'HAEMATOLOGY: MCHC']


george_labels_one_years = ['Age At Episode', 'GENDER',
     #  'COGNITIVE EXAM 120-161: COGNITIVE EXAM 120-161',
       'COGNITIVE EXAM 120-161: (121) IDENTIFIES DATE',
       'COGNITIVE EXAM 120-161: (144) REPETITION',
       'COGNITIVE EXAM 120-161: (160) SUBTRACTING SEVENS',
       'COGNITIVE EXAM 120-161: (161) RECALLS OBJECTS',
       'COGNITIVE EXAM 120-161: (161) RECALLS OBJECTS 3: PENNY',
     #  'COGNITIVE EXAM 162-187: COGNITIVE EXAM 162-187',
       'DIAGNOSIS 334-351: DEPRESSIVE ILLNESS',
       'DIAGNOSIS 334-351: ANXIETY/PHOBIC', 'CAMDEX SCORES: MINI MENTAL SCORE',
       'OPTIMA DIAGNOSES V 2010: CERBRO-VASCULAR DISEASE PRESENT']
       # 'OPTIMA DIAGNOSES V 2010: DIAGNOSTIC CODE' ]


score_dict = {'CAMDEX SCORES: CAMCOG EXPRESSION SCORE':(0,21), 'CAMDEX SCORES: CAMCOG LEARNING MEMORY SCORE':(0,27), 'CAMDEX SCORES: MINI MENTAL SCORE':(0,30), 'CAMDEX SCORES: TOTAL CAMCOG SCORE':(0,104), 
              'CAMDEX SCORES: CAMCOG ATTENTION SCORE':(0,9), 'CAMDEX SCORES: CAMCOG REMOTE MEMORY SCORE': (0,6), 'CAMDEX SCORES: CAMCOG RECENT MEMORY SCORE':(0,0)}


score_columns = ['CAMDEX SCORES: CAMCOG REMOTE MEMORY SCORE',
                 'CAMDEX SCORES: TOTAL CAMCOG SCORE',
                 'CAMDEX SCORES: CAMCOG EXPRESSION SCORE',
                 'CAMDEX SCORES: CAMCOG RECENT MEMORY SCORE',
                 'CAMDEX SCORES: CAMCOG ATTENTION SCORE',
                 'CAMDEX SCORES: CAMCOG LEARNING MEMORY SCORE',
                 'CAMDEX SCORES: MINI MENTAL SCORE']