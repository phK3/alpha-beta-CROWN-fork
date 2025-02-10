
import onnx
from onnx2pytorch import ConvertModel
import torch
import torch.nn as nn
import numpy as np

from collections import OrderedDict

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm


class MaxNN(nn.Module):
    def __init__(self):
        super(MaxNN, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)

    def forward(self, x):
        x = x.flatten(1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        y  = torch.max(x1, x2)
        return y
    

def make_maxpool_2layer_model(bound_opts=None):
    W1 = torch.tensor([[[[1., 1.],
                        [-1., 1.]]],
                      [[[-1., 0.],
                        [1., 1.]]]])
    W2 = -torch.ones(2, 2, 1, 1)
    b1 = torch.zeros(2)
    b2 = torch.zeros(2)

    net = nn.Sequential(nn.Conv2d(1, 2, 2, padding=1), nn.MaxPool2d(2), nn.Conv2d(2, 2, 1, padding=1), nn.MaxPool2d(2), nn.Flatten())

    with torch.no_grad():
        net[0].weight = nn.Parameter(W1)
        net[0].bias   = nn.Parameter(b1)
        net[2].weight = nn.Parameter(W2)
        net[2].bias   = nn.Parameter(b2)

    x = torch.zeros(1, 1, 2, 2)
    model = BoundedModule(net, x, bound_opts=bound_opts)    

    return model



def load_onnx_model(onnx_path, input_shape, bound_opts=None):
    onnx_model = onnx.load(onnx_path)
    torch_model = ConvertModel(onnx_model)
    
    x_concrete = torch.zeros(input_shape)
    model = BoundedModule(torch_model, x_concrete, bound_opts=bound_opts)
    return model


def get_layers(model):
    return [l for l in model.nodes() if l.perturbed]


def get_intermediate_bounds(model):
    """
    Returns a dictionary containing the concrete lower and upper bounds of each layer.
    
    Implemented own method to filter out bounds for weight matrices.
    
    Only call this method after compute_bounds()!
    """
    od = OrderedDict()
    for l in get_layers(model):
        if hasattr(l, 'lower'):
            od[l.name] = (l.lower, l.upper)
            
    return od


def check_bounds(model, xl, xu, yl, yu, n_test=100, print_only_summary=False):
    width = xu - xl

    cnt_lb = 0
    cnt_ub = 0
    for i in range(n_test):
        x = width * torch.rand_like(xl) + xl
        y = model(x)

        if not torch.all(y <= yu):
            if not print_only_summary:
                print("test failed: not a valid upper bound!")
                print("x = ", x)
                print("y = ", y)
            cnt_ub += 1
        
        if not torch.all(y >= yl):
            if not print_only_summary:
                print("test failed: not a valid lower bound!")
                print("x = ", x)
                print("y = ", y)
            cnt_lb += 1

    print(f"number of failures: {cnt_lb} (lower bound), {cnt_ub} (upper bound)")


def print_bounds(lbs, ubs, digits=3):
    lbs = lbs.detach().numpy()
    ubs = ubs.detach().numpy()
    with np.printoptions(precision=digits, suppress=True):
        print("lbs: ", lbs.reshape(-1))
        print("ubs: ", ubs.reshape(-1))



if __name__ == '__main__':
    run_cex = False
    run_small_example = True
    run_alpha = True
    run_large_model = False
    run_only_lp = False
    run_max_nn = False
    #large_model_path = '../vnncomp2021/benchmarks/verivital/Convnet_maxpool.onnx'
    #large_model_path = "../maxpool_networks/mnist_maxpool_small_2x2.onnx"
    large_model_path = "../maxpool_networks/mnist_maxpool_very_small_2x2.onnx"
    small_example_path = '../maxpool_example/onnx/maxpool_example.onnx'
    # TODO: why is it possible to convert this to pytorch, but execution with concrete input fails?
    #small_example_path = '../maxpool_example/onnx/maxpool_example_2layer.onnx'


    if run_max_nn:
        maxnet = MaxNN()
    
        data_min = -torch.ones(1, 1, 1, 2)
        data_max = torch.ones(1, 1, 1, 2)
        center = 0.5 * (data_min + data_max)
        ptb = PerturbationLpNorm(x_L=data_min, x_U=data_max)
        x = BoundedTensor(center, ptb)

        model = BoundedModule(maxnet, center)

        lbs, ubs = model.compute_bounds(x=(x,), method='crown')
        print_bounds(lbs, ubs)

        lbs, ubs = model.compute_bounds(x=(x,), method='alpha-crown')
        print_bounds(lbs, ubs)


    if run_cex:
        # not sure anymore, if this is a counterexample ...
        # is just their cnn-cert algorithm wrong?

        # cexnet has two inputs and just computes their max
        cexnet = nn.Sequential(nn.Conv2d(1, 1, (1, 2), padding=(0, 1), stride=2), nn.MaxPool2d((1, 2)), nn.Flatten())

        with torch.no_grad():
            W = torch.zeros(1, 1, 1, 2)
            W = torch.tensor([[[[1., 1.]]]])
            b = torch.zeros(1)
            
            cexnet[0].weight = nn.Parameter(W)
            cexnet[0].bias   = nn.Parameter(b)

        data_min = torch.zeros(1, 1, 1, 2)
        data_max = torch.tensor([0., 1.]).reshape(1, 1, 1, 2)
        center = 0.5 * (data_min + data_max)
        ptb = PerturbationLpNorm(x_L=data_min, x_U=data_max)
        x = BoundedTensor(center, ptb)

        model = BoundedModule(cexnet, center, bound_opts={'maxpool_relaxation': 'xiao2024_original'})
        print("## Xiao2024_original")
        lbs, ubs = model.compute_bounds(x=(x,), method='crown')
        print_bounds(lbs, ubs)

        check_bounds(model, data_min, data_max, lbs, ubs)

        model = BoundedModule(cexnet, center, bound_opts={'maxpool_relaxation': 'xiao2024'})
        print("\n## Xiao2024 (own)")
        lbs, ubs = model.compute_bounds(x=(x,), method='crown')
        print_bounds(lbs, ubs)

        check_bounds(model, data_min, data_max, lbs, ubs)

    if run_small_example:
        input_shape = (1, 1, 2, 2)
        #model = load_onnx_model(small_example_path, input_shape)
        model = make_maxpool_2layer_model(bound_opts={'optimize_bound_args': {'iteration': 5}})

        data_min = -torch.ones(4).reshape(input_shape)
        data_max = torch.ones(4).reshape(input_shape)
        center = 0.5 * (data_min + data_max)
        ptb = PerturbationLpNorm(x_L=data_min, x_U=data_max)
        x = BoundedTensor(center, ptb)
        
        print("## CROWN")
        lbs, ubs = model.compute_bounds(x=(x,), method='crown')
        print_bounds(lbs, ubs)

        check_bounds(model, data_min, data_max, lbs, ubs)

        if run_alpha:
            print("\n## aCROWN")
            lbs, ubs = model.compute_bounds(x=(x,), method='alpha-crown')
            print_bounds(lbs, ubs)

            check_bounds(model, data_min, data_max, lbs, ubs)

        if run_alpha or run_only_lp:
            # the first lr_alpha is useless, need the one in optimize_bound_args
            bound_opts = {'maxpool_relaxation': 'gurobi_lp', 'verbosity': 3, 'lr_alpha': 0.1, 'optimize_bound_args': {'lr_alpha': 0.1, 'iteration': 5}}
            #model = load_onnx_model(small_example_path, input_shape, bound_opts=bound_opts)
            model = make_maxpool_2layer_model(bound_opts=bound_opts)
            print("\n## aCROWN (gurobi_lp)")
            lbs, ubs = model.compute_bounds(x=(x,), method='alpha-crown')
            print_bounds(lbs, ubs)

            check_bounds(model, data_min, data_max, lbs, ubs)

        model = load_onnx_model(small_example_path, input_shape, bound_opts={'maxpool_relaxation': 'deeppoly'})
        print("\n## CROWN (refactored)")
        lbs, ubs = model.compute_bounds(x=(x,), method='crown')
        print_bounds(lbs, ubs)

        check_bounds(model, data_min, data_max, lbs, ubs)

        #model = load_onnx_model(small_example_path, input_shape, bound_opts={'maxpool_relaxation': 'xiao2024'})
        model = make_maxpool_2layer_model(bound_opts={'maxpool_relaxation': 'xiao2024'})
        print("\n## CROWN (xiao2024)")
        lbs, ubs = model.compute_bounds(x=(x,), method='crown')
        print_bounds(lbs, ubs)

        check_bounds(model, data_min, data_max, lbs, ubs)

        #model = load_onnx_model(small_example_path, input_shape, bound_opts={'maxpool_relaxation': 'xiao2024_original'})
        model = make_maxpool_2layer_model(bound_opts={'maxpool_relaxation': 'xiao2024_original'})
        print("\n## CROWN (xiao2024_original)")
        lbs, ubs = model.compute_bounds(x=(x,), method='crown')
        print_bounds(lbs, ubs)

        check_bounds(model, data_min, data_max, lbs, ubs)

        

    if run_large_model:
        print("\n## Testing larger model")
        input_shape = (1, 1, 28, 28)
        eps = 0.01
        X = np.load('./complete_verifier/datasets/eran/mnist_eran/X_eran.npy')
        center = torch.tensor(X[0:1,:,:,:])
        data_min = torch.clamp(center - eps, 0, 1)
        data_max = torch.clamp(center + eps, 0, 1)
        ptb = PerturbationLpNorm(x_L=data_min, x_U=data_max)
        x = BoundedTensor(center, ptb)


        model = load_onnx_model(large_model_path, input_shape, bound_opts={'optimize_bound_args': {'iteration': 5}})
        print("\n## CROWN")
        lbs, ubs = model.compute_bounds(x=(x,), method='crown')
        print_bounds(lbs, ubs)

        check_bounds(model, data_min, data_max, lbs, ubs)

        if run_alpha:
            print("\n## aCROWN")
            lbs, ubs = model.compute_bounds(x=(x,), method='alpha-crown')
            print_bounds(lbs, ubs)

            check_bounds(model, data_min, data_max, lbs, ubs)

        model = load_onnx_model(large_model_path, input_shape, bound_opts={'maxpool_relaxation': 'deeppoly', 'optimize_bound_args': {'iteration': 5}})
        print("\n## CROWN (refactored)")
        lbs, ubs = model.compute_bounds(x=(x,), method='crown')
        print_bounds(lbs, ubs)

        check_bounds(model, data_min, data_max, lbs, ubs)

        if run_alpha:
            print("\n## aCROWN (refactored)")
            lbs, ubs = model.compute_bounds(x=(x,), method='alpha-crown')
            print_bounds(lbs, ubs)

            check_bounds(model, data_min, data_max, lbs, ubs)

        model = load_onnx_model(large_model_path, input_shape, bound_opts={'maxpool_relaxation': 'xiao2024', 'verbosity': 3, 'optimize_bound_args': {'iteration': 5}})
        print("\n## CROWN (xiao2024)")
        lbs, ubs = model.compute_bounds(x=(x,), method='crown')
        print_bounds(lbs, ubs)

        check_bounds(model, data_min, data_max, lbs, ubs)

        if run_alpha:
            print("\n## aCROWN (xiao2024)")
            lbs, ubs = model.compute_bounds(x=(x,), method='alpha-crown')
            print_bounds(lbs, ubs)

            check_bounds(model, data_min, data_max, lbs, ubs)

        model = load_onnx_model(large_model_path, input_shape, bound_opts={'maxpool_relaxation': 'xiao2024_original'})
        print("\n## CROWN (xiao2024_original)")
        lbs, ubs = model.compute_bounds(x=(x,), method='crown')
        print_bounds(lbs, ubs)

        check_bounds(model, data_min, data_max, lbs, ubs)

        if run_alpha or run_only_lp:
            # the first lr_alpha is useless, need the one in optimize_bound_args
            bound_opts = {'maxpool_relaxation': 'gurobi_lp', 'verbosity': 3, 'lr_alpha': 0.1, 'optimize_bound_args': {'lr_alpha': 0.1, 'iteration': 5}}
            model = load_onnx_model(large_model_path, input_shape, bound_opts=bound_opts)
            print("\n## aCROWN (gurobi_lp)")
            lbs, ubs = model.compute_bounds(x=(x,), method='alpha-crown')
            print_bounds(lbs, ubs)

            check_bounds(model, data_min, data_max, lbs, ubs, print_only_summary=True)

    print('yay')





