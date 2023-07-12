#    pretrained-vit-pytorch

  This GitHub repository contains a pretrained Vit model that described in the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale". The paper proposes a Transformer model with an attention mechanism for large-scale image recognition problems. The model was implemented in Python using the PyTorch library.

#    About ViT

![alt text](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vit_architecture.jpg)

#    Usage

<code>pretrained_vit_results = engine.train(model=pretrained_vit,
                                          train_dataloader=train_dataloader_pretrained,
                                          test_dataloader=test_dataloader_pretrained,
                                          optimizer=optimizer,
                                          loss_fn=loss_fn,
                                          epochs=EPOCHS,
                                          device=device)</code>
