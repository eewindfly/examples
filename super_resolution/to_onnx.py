import torch

torch_model =  torch.load('./models/model_epoch_100.pth')
torch_model.eval()

batch_size = 1    # just a random number

x = torch.randn(batch_size, 1, 540, 960, requires_grad=True)
x = x.cuda()

torch_out = torch_model(x)
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "./models/espcn.onnx",              # where to save the model (can be a file or file-like        object)
                  export_params=True,        # store the trained parameter weights inside the   model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for          optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  verbose=True
                  )

print("Finish!")
