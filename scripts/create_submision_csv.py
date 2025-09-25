import pandas as pd
import os

def main():
    """
    Creates a CSV file listing the filenames of predicted masks for submission.

    This is a simple helper script to generate a submission file based on the
    contents of a directory of predicted label masks.
    """
    dir_path = '/path/to/labels'
    
    image_names = [image_name.split('.png')[0] for image_name in os.listdir(dir_path)]
    
    csv = pd.DataFrame(image_names, columns=['filename'])
    csv.to_csv('/path_to/output.csv', index=False)
    print(f"Submission CSV created with {len(image_names)} entries.")

if __name__ == "__main__":
    main()
