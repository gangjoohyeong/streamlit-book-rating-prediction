import torch
import streamlit as st
from model import FactorizationMachineModel
from data_load import context_data_load
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, Dataset


def load_model(test_ratings:pd.DataFrame) -> FactorizationMachineModel:
    data = context_data_load(test_ratings)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FactorizationMachineModel(data).to(device)
    model.load_state_dict(torch.load('./model_file/FM_model.pt', map_location=device))

    return data, model
    
def get_prediction(model:FactorizationMachineModel, data):
    batch_size = 1024
    predicts = list()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    for data in test_dataloader:
        x = data[0].to(device)
        y_hat = model(x)
        predicts.extend(y_hat.tolist())
    return predicts