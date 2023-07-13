#    pretrained-vit-pytorch

  This GitHub repository contains a pretrained Vit model. Original model described in the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale". The paper proposes a Transformer model with an attention mechanism for large-scale image recognition problems. The model was implemented in Python using the PyTorch library.

#    About ViT

![alt text](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vit_architecture.jpg)

#    Usage

<code>EPOCH: int
</code>You can write the value of how many EPOCHs you want to train the model.

<code>batch_size: int
</code>The value should be 8 or its multiples. You can increase or decrease it based on the size of your data.




<pre>    #This part is responsible for training. If you have already trained your model, you can comment out this part.   
  
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

