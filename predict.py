import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from tqdm import tqdm
from PIL import Image
from glob import glob

import network
from datasets import PaintedSymbols
from torchvision import transforms as T

def get_argparser():
    parser = argparse.ArgumentParser(description="Predict segmentation masks for road markings")

    parser.add_argument("--input", type=str, required=True, help="Path to a single image or an image directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output masks.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the trained model checkpoint.")
    
    # Model Options
    available_models = sorted(
        name for name in network.modeling.__dict__
        if name.islower() and not name.startswith("__") and callable(network.modeling.__dict__[name])
    )
    parser.add_argument("--model", type=str, default="deeplabv3plus_resnet101", choices=available_models, help="Model architecture.")
    parser.add_argument("--separable_conv", action='store_true', default=False, help="Apply separable conv to decoder and aspp.")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16], help="Output stride for DeepLab.")
    
    # Prediction Options
    parser.add_argument("--output_type", type=str, default='label', choices=['label', 'color'], help="Type of mask to save: 'label' for competition format (0,34,41), 'color' for visualization.")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use.")

    return parser

def main():
    opts = get_argparser().parse_args()
    os.makedirs(opts.output_dir, exist_ok=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Setup model
    model = network.modeling.__dict__[opts.model](num_classes=3, output_stride=opts.output_stride)
    if opts.separable_conv:
        network.convert_to_separable_conv(model.classifier)
    
    checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state'])
    model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, f'**/*.{ext}'), recursive=True)
            image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)
    
    print(f"Found {len(image_files)} images to predict.")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Predicting"):
            img_name = os.path.basename(img_path)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            pred = model(img_tensor).max(1)[1].cpu().numpy()[0] # HW, values are 0, 1, 2

            if opts.output_type == 'label':
                # Remap train IDs (0, 1, 2) to original label IDs (0, 34, 41)
                output_mask = np.zeros_like(pred, dtype=np.uint8)
                output_mask[pred == 1] = 34  # crosswalk
                output_mask[pred == 2] = 41  # restricted_area
                mask_img = Image.fromarray(output_mask)
            else: # 'color'
                # Decode to color image for visualization
                colorized_mask = PaintedSymbols.decode_target(pred).astype('uint8')
                mask_img = Image.fromarray(colorized_mask)
            
            mask_img.save(os.path.join(opts.output_dir, img_name))

    print(f"Predictions saved to {opts.output_dir}")

if __name__ == '__main__':
    main()