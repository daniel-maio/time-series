import torch
import pandas


def get_anomaly_readings(train_errors, errors, original_df, std = 2) -> pandas.DataFrame:

    '''
    Return the readings with anomaly.

    Arg:
    
    train_erros: MSE tensor of the training.
    
    errors: MSE of the current data (eval).
    
    original_df: current data in pandas.Dataframe format.

    Return:

    Rows of the current data with anomaly.
    '''

    if len(errors.size()) < 3:
        batch_size = 1
    else:
        batch_size = errors.size(0)

    std_ = torch.std(train_errors, dim=0).unsqueeze(0).repeat(batch_size,1,1)

    anomaly_mask = torch.sqrt(errors) > std * std_

    i, j = torch.nonzero(anomaly_mask, as_tuple=True)[:-1]

    idx = torch.unique(i + j)

    return original_df.iloc[idx]