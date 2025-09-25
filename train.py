import torch
import torch.nn as nn
import numpy as np
import random
import os
import argparse
from tqdm import tqdm
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

import network
import scripts
from scripts import ext_transforms as et
from metrics import StreamSegMetrics
from datasets import PaintedSymbols
from torch.scripts import data
from scripts.visualizer import Visualizer

def get_argparser():
    parser = argparse.ArgumentParser(description="DeepLabV3+ for Road Marking Segmentation")

    parser.add_argument("--dataset_csv", type=str, required=True, help="Path to the master CSV file for the dataset.")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of classes (background, crosswalk, restricted_area).")
    
    available_models = sorted(
        name for name in network.modeling.__dict__
        if name.islower() and not name.startswith("__") and callable(network.modeling.__dict__[name])
    )
    parser.add_argument("--model", type=str, default="deeplabv3plus_resnet101", choices=available_models, help="Model architecture.")
    parser.add_argument("--separable_conv", action='store_true', default=False, help="Apply separable conv to decoder and aspp.")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16], help="Output stride for DeepLab.")
    parser.add_argument("--pretrained_on", type=str, default='cityscapes', choices=['cityscapes', 'imagenet', None], help="What weights to use for backbone initialization.")

    parser.add_argument("--total_itrs", type=int, default=50e3, help="Number of training iterations.")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("--lr_policy", type=str, default="poly", choices=["poly", "step"], help="Learning rate scheduler policy.")
    parser.add_argument("--step_size", type=int, default=10000, help="Step size for 'step' LR policy.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--val_batch_size", type=int, default=4, help="Validation batch size.")
    parser.add_argument("--crop_size", type=int, default=768, help="Image crop size for training.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed.")
    
    parser.add_argument("--ckpt", default=None, type=str, help="Path to checkpoint to restore from.")
    parser.add_argument("--continue_training", action='store_true', default=False, help="Restore training state from checkpoint.")
    
    parser.add_argument("--val_interval", type=int, default=500, help="Iteration interval for validation.")
    parser.add_argument("--save_val_results", action='store_true', default=False, help="Save validation results to disk.")
    parser.add_argument("--print_interval", type=int, default=10, help="Iteration interval for printing loss.")
    parser.add_argument("--enable_vis", action='store_true', default=False, help="Enable visdom visualization.")
    parser.add_argument("--vis_port", type=int, default=8097, help="Port for visdom server.")
    parser.add_argument("--vis_env", type=str, default="main", help="Environment for visdom.")
    parser.add_argument("--vis_num_samples", type=int, default=8, help="Number of samples to visualize.")

    return parser

def get_dataset(opts):
    """ Configure dataloaders """
    train_transform = et.ExtCompose([
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
        et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dst = PaintedSymbols(csv_path=opts.dataset_csv, split='train', transform=train_transform)
    val_dst = PaintedSymbols(csv_path=opts.dataset_csv, split='eval', transform=val_transform)
    return train_dst, val_dst

def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        os.makedirs('results', exist_ok=True)
        denorm = scripts.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_id = 0

    model.eval()
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader), total=len(loader), desc="Validating"):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:
                ret_samples.append((images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for j in range(len(images)):
                    image = (denorm(images[j].detach().cpu().numpy()) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(targets[j]).astype(np.uint8)
                    pred = loader.dataset.decode_target(preds[j]).astype(np.uint8)

                    Image.fromarray(image).save(f'results/{img_id}_image.png')
                    Image.fromarray(target).save(f'results/{img_id}_target.png')
                    Image.fromarray(pred).save(f'results/{img_id}_pred.png')
                    img_id += 1
    
    score = metrics.get_results()
    return score, ret_samples

def main():
    opts = get_argparser().parse_args()

    vis = Visualizer(port=opts.vis_port, env=opts.vis_env) if opts.enable_vis else None
    if vis:
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = data.DataLoader(val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=4)
    print(f"Dataset loaded: Train set: {len(train_dst)}, Val set: {len(val_dst)}")

    num_classes_pretrained = 19 if opts.pretrained_on == 'cityscapes' else opts.num_classes
    model = network.modeling.__dict__[opts.model](num_classes=num_classes_pretrained, output_stride=opts.output_stride)
    
    if opts.separable_conv:
        network.convert_to_separable_conv(model.classifier)
    scripts.set_bn_momentum(model.backbone, momentum=0.01)

    if opts.pretrained_on == 'cityscapes':
        last_conv_in_channels = model.classifier.classifier[-1].in_channels
        model.classifier.classifier[-1] = nn.Conv2d(last_conv_in_channels, opts.num_classes, kernel_size=1)
        print("Replaced model's final layer for 3-class segmentation.")

    optimizer = torch.optim.SGD(
        params=[
            {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
            {'params': model.classifier.parameters(), 'lr': opts.lr},
        ],
        lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay
    )

    if opts.lr_policy == 'poly':
        scheduler = scripts.PolyLR(optimizer, opts.total_itrs, power=0.9)
    else: # step
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    metrics = StreamSegMetrics(opts.num_classes)
    
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored.")
        print(f"Model restored from {opts.ckpt}")
        del checkpoint
    else:
        print("No checkpoint found, training from scratch...")
        model = nn.DataParallel(model)
        model.to(device)
        
    def save_ckpt(path):
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print(f"Model saved to {path}")

    interval_loss = 0
    while cur_itrs < opts.total_itrs:
        model.train()
        cur_epochs += 1
        for images, labels in train_loader:
            cur_itrs += 1
            if cur_itrs > opts.total_itrs: break

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            np_loss = loss.item()
            interval_loss += np_loss
            if vis: vis.vis_scalar('Loss', cur_itrs, np_loss)
            
            if (cur_itrs) % opts.print_interval == 0:
                interval_loss /= opts.print_interval
                print(f"Epoch {cur_epochs}, Itrs {cur_itrs}/{opts.total_itrs}, Loss={interval_loss:.4f}")
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                print("Validating...")
                val_score, _ = validate(opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)
                print(metrics.to_str(val_score))

                if val_score['Mean IoU'] > best_score:
                    best_score = val_score['Mean IoU']
                    save_ckpt(f'checkpoints/best_{opts.model}.pth')

                if vis:
                    vis.vis_scalar('[Val] Mean IoU', cur_itrs, val_score['Mean IoU'])

    print("Training finished.")

if __name__ == '__main__':
    main()