import torch
import torchvision
from torch import nn
from modules import data_setup, engine, utils, predictions

device = "cuda" if torch.cuda.is_available() else "cpu"
image_path = "img"
train_dir = image_path + "/train"
test_dir = image_path + "/test"
EPOCHS = 10
BATCH_SIZE = 32
if __name__ == '__main__':

    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False
    pretrained_vit_transforms = pretrained_vit_weights.transforms()
    train_dataloader_pretrained, test_dataloader_pretrained, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=pretrained_vit_transforms,
        batch_size=BATCH_SIZE)
    pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)

    #    This part is responsible for training. If you have already trained your model, you can comment out this part.   
    
    optimizer = torch.optim.Adam(params=pretrained_vit.parameters(),
                                 lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    pretrained_vit_results = engine.train(model=pretrained_vit,
                                          train_dataloader=train_dataloader_pretrained,
                                          test_dataloader=test_dataloader_pretrained,
                                          optimizer=optimizer,
                                          loss_fn=loss_fn,
                                          epochs=EPOCHS,
                                          device=device)
    
    utils.plot_loss_curves(pretrained_vit_results)

    #    Saves model to choosen directory with choosen name.
    
    utils.save_model(model=pretrained_vit,
                     target_dir="models",
                     model_name="vitmodel.pth")
    
    #    Loads model from choosen directory.
    
    pretrained_vit.load_state_dict(torch.load("models/vitmodel.pth"))
    pretrained_vit.eval()

    
    custom_image_path = image_path + "/47.jpg"
    #    Predicts the image
    
    predictions.pred_and_plot_image(model=pretrained_vit,
                                    image_path=custom_image_path,
                                    class_names=class_names)

