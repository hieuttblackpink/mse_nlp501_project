import numpy as np
from termcolor import colored

from keras.layers import Embedding
from keras.layers import GRU
from keras.layers import Dense 

# Compare the two inputs
def comparator(learner, instructor):
    if len(learner) != len(instructor):
        raise AssertionError(f"The number of layers in the proposed model does not agree with the expected model: expected {len(instructor)}, got {len(learner)}.") 
    for a, b in zip(learner, instructor):
        if tuple(a) != tuple(b):
            print(colored("Test failed", attrs=['bold']),
                  "\n Expected value \n\n", colored(f"{b}", "green"), 
                  "\n\n does not match the input value: \n\n", 
                  colored(f"{a}", "red"))
            raise AssertionError("Error in test") 
    print(colored("All tests passed!", "green"))

# extracts the description of a given model
# def summary(model):
#     result = []
#     for layer in model.layers:
#         descriptors = [layer.__class__.__name__, layer.output_shape, layer.count_params()]
#         if (type(layer) == Dense):
#             descriptors.append(layer.activation.__name__.replace("_v2", ""))
#         if (type(layer) == GRU):
#             descriptors.append(f"return_sequences={layer.return_sequences}")
#             descriptors.append(f"return_state={layer.return_state}")
#         result.append(descriptors)
#     return result

def summary(model):
    result = []
    for layer in model.layers:
        # Try to get output shape safely
        try:
            output_shape = layer.output_shape
        except AttributeError:
            try:
                output_shape = layer.output.shape
            except Exception:
                output_shape = None  # fallback if not available

        descriptors = [layer.__class__.__name__, output_shape, layer.count_params()]

        if isinstance(layer, Dense):
            descriptors.append(layer.activation.__name__.replace("_v2", ""))
        if isinstance(layer, GRU):
            descriptors.append(f"return_sequences={layer.return_sequences}")
            descriptors.append(f"return_state={layer.return_state}")

        result.append(descriptors)
    return result
