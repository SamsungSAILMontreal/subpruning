import inspect
import os
from functools import partial

import numpy as np
import json

from sklearn.pipeline import Pipeline 

# import cloudpickle
# # TODO Change function header to be nicer
# def store_model(previous_output, model, out_path = None, x_train = None, y_train = None, x_test = None, y_test = None):
#     print("IN STORE MODEl")
#     if hasattr(model, "store"):
#         print("FOUND CUSTOM MODEL")
#         # Custom model
#         # TODO Remove dim options here
#         model.store("{}".format(out_path), dim=x_train[0].shape, name="model") 
#     else:
#         if isinstance(model, Pipeline) and hasattr(model.steps[-1], "store"):
#             print("FOUND PIPELINE WITH CUSTOM MODEL")
#             # SKLEARN PIPELINE With custom model
#             custom_model = model.steps.pop()
#             with open(os.path.join(out_path, "model_pipeline.pkl"), "wb") as f:
#                 cloudpickle.dump(model, f)

#             # TODO Remove dim options here
#             custom_model.store("{}".format(out_path), dim=x_train[0].shape, name="model")
#         else:
#             print("FOUND PIPELINE")
#             print("ADWDW")
#             with open(os.path.join(out_path, "model.pkl"), "wb") as f:
#                 print("PICKLE WITH CLOUDPICKLE")
#                 cloudpickle.dump(model, f)
#             # SKLEARN MODEL OR PIPELINE

def replace_objects(d):
    d = d.copy()
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = replace_objects(v)
        elif isinstance(v, list):
            d[k] = [replace_objects({"key":vv})["key"] for vv in v]
        elif isinstance(v, np.generic):
            d[k] = v.item()
        elif isinstance(v, partial):
            d[k] = v.func.__name__ + "_" + "_".join([str(arg) for arg in v.args]) + str(replace_objects(v.keywords))
        elif callable(v) or inspect.isclass(v):
            try:
                d[k] = v.__name__
            except:
                d[k] = str(v) #.__name__
        elif isinstance(v, object) and v.__class__.__module__ != 'builtins':
            # print(type(v))
            d[k] = str(v)
        else:
            d[k] = v
    return d        

def stacktrace(exception):
    """convenience method for java-style stack trace error messages"""
    import sys
    import traceback
    print("\n".join(traceback.format_exception(None, exception, exception.__traceback__)),
        #file=sys.stderr,
        flush=True)

def cfg_to_str(cfg):
    cfg = replace_objects(cfg.copy())
    return json.dumps(cfg, indent=4)    

# getfullargspec does not handle inheritance correctly.
# Taken from https://stackoverflow.com/questions/36994217/retrieving-arguments-from-a-class-with-multiple-inheritance
def get_ctor_arguments(clazz):
    args = ['self']
    for C in clazz.__mro__:
        if '__init__' in C.__dict__:
            args += inspect.getfullargspec(C).args[1:]
            args += inspect.getfullargspec(C).kwonlyargs
    return args