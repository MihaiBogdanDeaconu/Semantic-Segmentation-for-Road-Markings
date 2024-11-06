import pandas as pd
import os

dir_path = '/shares/CC_v_Val_FV_Gen3_all/SemsegCrosswalkRestrictedArea/mihai/labels'

image_names = [image_name.split('.png')[0] for image_name in os.listdir(dir_path)]

csv = pd.DataFrame(image_names, columns=['filename'])

csv.to_csv('/shares/CC_v_Val_FV_Gen3_all/SemsegCrosswalkRestrictedArea/mihai/MT-DNN_Heads_output.csv', index = False)