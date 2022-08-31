from tabpy.tabpy_tools.client import Client
import pandas as pd
import numpy as np
import pickle


model = pickle.load(open('rf.pkl', 'rb'))

pre_process = pickle.load(open('pre_process.pkl', 'rb'))


def Hospitalization_Prediction(_arg1, _arg2, _arg3, _arg4, _arg5, _arg6, _arg7, _arg8):

    input_data = np.column_stack([_arg1, _arg2, _arg3, _arg4, _arg5, _arg6, _arg7, _arg8])
    X = pd.DataFrame(input_data,columns=['age','MAP','triage_acuity','triage_heartrate','triage_temperature','triage_resprate','triage_o2sat','n_ed_365d'])
    result = model.predict_proba(pre_process.transform(X))
    return [probability[1] for probability in result]


client = Client('http://localhost:9004/')

# Connect to TabPy server using the client library
connection = Client('http://localhost:9004/')

connection.deploy('Hospitalization_Prediction', 
Hospitalization_Prediction,
'Returns prediction of probability of Hospitalization.',override=True)