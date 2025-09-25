import pandas as pd
import os

dir_path = '/path/to/labels'

image_names = [image_name.split('.png')[0] for image_name in os.listdir(dir_path)]

csv = pd.DataFrame(image_names, columns=['filename'])

csv.to_csv('/path_to/output.csv', index = False)