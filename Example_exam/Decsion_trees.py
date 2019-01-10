import numpy as np
import pandas as pd

quinlan_data = pd.read_table('quinlan_data.txt',
                             sep=' ',
                             header=None,
                             names=['Outlook', 'Temperature', 'Humidity', 'Wind', 'Class']
                             )

# print(quinlan_data[quinlan_data['Outlook'] == 'Rain' and quinlan_data['Wind'] == 'False'])

print(quinlan_data[quinlan_data['Outlook'] == 'Sunny'])


