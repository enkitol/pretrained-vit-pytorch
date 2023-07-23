#    pretrained-vit-pytorch

  This GitHub repository contains a pretrained Vit model, you can use it for image classification. Original model described in the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale". The paper proposes a Transformer model with an attention mechanism for large-scale image recognition problems. The model was implemented in Python using the PyTorch library.

#    ViT Model

![alt text](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vit_architecture.jpg)

#    Usage

<code>EPOCH: int
</code>You can change the value of how many EPOCHs you want to train the model.

<code>BATCH_SIZE: int
</code>This value should be 8 or its multiples. You can increase or decrease it based on the size of your data.




<pre>    #  This part is responsible for training. If you have already trained your model, you can comment out this part.
  
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
    utils.save_model(model=pretrained_vit,
                     target_dir="models",
                     model_name="vitmodel.pth")</pre>
<pre>    #  Saves model to choosen directory with choosen name. If you have trained model, you can comment out this part and load that trained model instead.
  
    utils.save_model(model=pretrained_vit,
                     target_dir="models",
                     model_name="vitmodel.pth")
</pre>

<pre>    #    Loads model from choosen directory.
    
    pretrained_vit.load_state_dict(torch.load("models/vitmodel.pth"))
    pretrained_vit.eval()
</pre>

<pre>    #    Predicts the image
    
    predictions.pred_and_plot_image(model=pretrained_vit,
                                    image_path=custom_image_path,
                                    class_names=class_names)</pre>
