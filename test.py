import os
import torch
from dataset_loader import CAMUSDataset, RandomGenerator
from networks.vit_seg_modelling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import evaluate

def main():
    # Configurations
    model_path = "./output/best_model.pth"  # Update to the correct model path
    vit_model_name = "R50-ViT-B_16"  # Example
    output_size = (224, 224)
    batch_size = 16
    data_dir = "preprocessed_data"  # Update this path
    num_epochs = 100
    num_classes = 4
    img_size = 224
    vit_patches_size = 16

    # Load Test Dataset
    test_dataset = CAMUSDataset(data_dir, split="test", transform=RandomGenerator(output_size))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load Model
    config_vit = CONFIGS_ViT_seg[vit_model_name]
    config_vit.n_classes = num_classes
    if vit_model_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))  
    #net.load_from(weights=np.load(config_vit.pretrained_path))
    #model = ViT_seg(img_size=224, num_classes=4, vit_name=vit_model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViT_seg(config_vit, img_size=224, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Evaluation
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_dice, test_iou, test_hausdorff_95 = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, DICE: {test_dice:.4f}, IoU: {test_iou:.4f}, Hausdorff95: {test_hausdorff_95:.4f}")

if __name__ == "__main__":
    main()
