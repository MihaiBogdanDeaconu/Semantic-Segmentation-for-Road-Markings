import pandas as pd 

def change_path_to_label_path(path):
    """
    Converts an image file path to its corresponding label file path.
    
    This function assumes a specific directory structure where images and labels
    are in parallel folders. For example:
    .../dataset/images/image_01.png -> .../dataset/splited_labels/train_targets/image_01.png

    Args:
        path (str): The file path of the input image.

    Returns:
        (str): The corresponding file path for the label mask.
    """
    image_name = path.split('/')[-1]
    path = path.split('/images')[0] + '/splited_labels/train_targets/' + image_name
    return path

def load_from_csv(csv_path, split):
    """
    Loads image and target file paths from a CSV for a specific data split.

    Args:
        csv_path (str): The path to the master CSV file.
        split (str): The desired data split, e.g., 'train' or 'eval'.

    Returns:
        A tuple of (list, list): 
        - A list of image file paths.
        - A list of corresponding label file paths.
    """
    df = pd.read_csv(csv_path)
    df_cur_split = df[df['split'] == split]
    images = df_cur_split['path'].tolist()
    targets = [change_path_to_label_path(path) for path in images]
    return images, targets
