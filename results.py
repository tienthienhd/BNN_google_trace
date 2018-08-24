# -*- coding: utf-8 -*-

from config import Config
from data import Data
from datetime import datetime
import os
import pandas as pd
import numpy as np


result_directory = ['./log/results/data_resource_usage_3Minutes_6176858948/config_0/encoder_decoder/',
                    './log/results/data_resource_usage_3Minutes_6176858948/config_8/encoder_decoder/',
                    './log/results/data_resource_usage_5Minutes_6176858948/config_1/encoder_decoder/',
                    './log/results/data_resource_usage_5Minutes_6176858948/config_9/encoder_decoder/',
                    './log/results/data_resource_usage_8Minutes_6176858948/config_2/encoder_decoder/',
                    './log/results/data_resource_usage_8Minutes_6176858948/config_10/encoder_decoder/',
                    './log/results/data_resource_usage_10Minutes_6176858948/config_3/encoder_decoder/',
                    './log/results/data_resource_usage_10Minutes_6176858948/config_11/encoder_decoder/'
                    ]


for directory in result_directory:        
    mae_rmse_result = pd.read_csv(directory + 'mae_rmse_log.csv', header=None).values
    
#    a = mae_rmse_result[:, 0]
    mae_min = np.amin(mae_rmse_result[:, 0])
    ind_mae_min = np.argmin(mae_rmse_result[:, 0])
    
    rmse_min = np.amin(mae_rmse_result[:, 1])
    ind_rmse_min = np.argmin(mae_rmse_result[:, 1])
    
    print('Config_{}:\n mae_min:{}-{}  rmse_min:{}-{}'.format(directory, mae_min, ind_mae_min, rmse_min, ind_rmse_min))
