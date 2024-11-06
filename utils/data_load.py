import pandas as pd 
def change_path_to_label_path(path):
    image_name = path.split('/')[-1]
    path = path.split('/images')[0] + '/splited_labels/train_targets/' + image_name
    return path

def load_from_csv(csv_path, split):

    df = pd.read_csv(csv_path)

    df_cur_split = df[df['split'] == split]

    images = df_cur_split['path'].tolist()
    targets = [change_path_to_label_path(path) for path in images]

    return images, targets