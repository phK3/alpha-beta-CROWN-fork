
import onnx
from onnx2pytorch import ConvertModel
import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm



class MaxPoolVerification_6Layers(nn.Module):

    def __init__(self, bias_model_layers):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = bias_model_layers[1]
        self.fc2 = bias_model_layers[3]
        self.fc3 = bias_model_layers[5]
        self.fc4 = bias_model_layers[7]
        self.fc5 = bias_model_layers[9]
        self.fc6 = bias_model_layers[11]
        self.fc7 = bias_model_layers[13]

    def forward(self, x):
        # l, u, alpha for all batches and for all of the w x h inputs
        lua = x[:,0:3,:,:]
        l = x[:,0:1,:,:]
        u = x[:,1:2,:,:]
        alpha = x[:,2:3,:,:]
        # normalized inputs to [0, 1] for all batches ans all w x h inputs
        x_in = x[:,3:4,:,:]

        # get bias prediction from model
        #b = self.bias_model(lua)
        b = b = self.flatten(lua)
        b = self.fc1(b)
        b = nn.functional.relu(b)
        b = self.fc2(b)
        b = nn.functional.relu(b)
        b = self.fc3(b)
        b = nn.functional.relu(b)
        b = self.fc4(b)
        b = nn.functional.relu(b)
        b = self.fc5(b)
        b = nn.functional.relu(b)
        b = self.fc6(b)
        b = nn.functional.relu(b)
        b = self.fc7(b)
        
        # transfrom input from [0, 1] to [l, u]
        x_hat = l + (u - l)*x_in 

        # we need alpha^T x_hat for all the batches 
        # so we manipulate the shapes to express it as matmul
        alpha_flat = alpha.flatten(-2)
        x_hat_flat = x_hat.flatten(-2)
        x_hat_T = x_hat_flat.transpose(-1, -2)
        # don't want squeeze because it depends on the dynamic size of the tensor
        # and only removes the last dimension if it has shape 1.
        # so ONNX may convert it to an if-statement for which abCROWN has not abstract transformer
        # since the last dim should always be 1, we just use [:,:,0] to remove it.
        #a_T_x = torch.matmul(alpha_flat, x_hat_T).squeeze(-1)
        a_T_x = torch.matmul(alpha_flat, x_hat_T)[:,:,0]
        y = a_T_x + b

        # we want alpha^T x_hat + b >= x_hat_i for all i
        # <==> 0 >= x_hat_i - (alpha^T x_hat  + b)
        # so we want the RHS to be smaller equal 0, if it is 
        # larger than 0, we have a violation!!!
        violation = x_hat.flatten(-3) - y 

        return violation
    


if __name__ == '__main__':
    onnx_model = onnx.load("./jupyter/models/net6x50_best.onnx")
    model = ConvertModel(onnx_model, experimental=False)
    mpv6 = MaxPoolVerification_6Layers(list(model.children()))

    x = torch.zeros(1, 4, 2, 2)
    print("mpv6(0) = ", mpv6(x))

    model_abcrown = BoundedModule(mpv6, x)

    data_min = torch.zeros_like(x)
    data_max = torch.ones_like(x)
    center = 0.5 * (data_min + data_max)

    ptb = PerturbationLpNorm(x_L=data_min, x_U=data_max)
    bound_x = BoundedTensor(center, ptb)

    lbs, ubs = model_abcrown.compute_bounds(x=(bound_x,), method='crown')
    print("lbs = ", lbs)
    print("ubs = ", ubs)

