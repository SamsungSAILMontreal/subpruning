import copy
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from functools import partial
import json
import os
import smtplib
import socket
import time
import traceback
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import ray
import random

from experiment_runner.Utils import stacktrace, cfg_to_str, get_ctor_arguments, replace_objects

def eval_fit(config):
    pre, fit, post, out_path, experiment_id, cfg = config
    try:
        # Make a copy of the model config for all output-related stuff
        # This does not include any fields which hurt the output (e.g. x_test,y_test)
        # but are usually part of the original modelcfg
        # if not verbose:
        #     import warnings
        #     warnings.filterwarnings('ignore')
        readable_cfg = copy.deepcopy(cfg)
        readable_cfg["experiment_id"] = experiment_id
        readable_cfg["out_path"] = out_path

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with open(out_path + "/config.json", 'w') as out:
            out.write(json.dumps(replace_objects(readable_cfg), indent=4))

        scores = {}
        repetitions = cfg.get("repetitions",1)
        for i in range(repetitions):
            if repetitions > 1:
                rep_out_path = os.path.join(out_path,str(i))
                if not os.path.exists(rep_out_path):
                    os.makedirs(rep_out_path)
            else:
                rep_out_path = out_path

            experiment_cfg = {
                **cfg,
                'experiment_id':experiment_id,
                'out_path':rep_out_path, 
                'run_id':i
            }

            if pre is not None:
                pre_stuff = pre(experiment_cfg)
                start_time = time.time()
                fit_stuff = fit(experiment_cfg, pre_stuff)
                fit_time = time.time() - start_time
            else:
                start_time = time.time()
                fit_stuff = fit(experiment_cfg)
                fit_time = time.time() - start_time
            
            if post is not None:
                cur_scores = post(experiment_cfg, fit_stuff)
                cur_scores["fit_time"] = fit_time

                if i == 0:
                    for k in list(cur_scores.keys()):
                        scores[k] = [cur_scores[k]]
                else:
                    for k in list(scores.keys()):
                        scores[k].append(cur_scores[k])

        for k in list(scores.keys()):
            scores["mean_" + k] = np.mean(scores[k])
            scores["std_" + k] = np.std(scores[k])

        readable_cfg["scores"] = scores
        out_file_content = json.dumps(replace_objects(readable_cfg), sort_keys=True) + "\n"
        
        return experiment_id, scores, out_file_content
    except Exception as identifier:
        stacktrace(identifier)
        # Ray is somtimes a little bit to quick in killing our processes if something bad happens 
        # In this case we do not see the stack trace which is super annyoing. Therefore, we sleep a
        # second to wait until the print has been processed / flushed
        time.sleep(1.0)
        return None

@ray.remote(max_calls=1)
def ray_eval_fit(pre, fit, post, out_path, experiment_id, cfg):
    return eval_fit( (pre, fit, post, out_path, experiment_id, cfg) )

def run_experiments(basecfg, cfgs, **kwargs):
    try:
        return_str = ""
        results = []
        if "out_path" in basecfg:
            basecfg["out_path"] = os.path.abspath(basecfg["out_path"])

        if not os.path.exists(basecfg["out_path"]):
            os.makedirs(basecfg["out_path"])
        else:
            if os.path.isfile(basecfg["out_path"] + "/results.jsonl"):
                os.unlink(basecfg["out_path"] + "/results.jsonl")

        # pool = NonDaemonPool(n_cores, initializer=init, initargs=(l,shared_list))
        # Lets use imap and not starmap to keep track of the progress
        # ray.init(address="ls8ws013:6379")
        backend = basecfg.get("backend", "local")
        verbose = basecfg.get("verbose", True)
        print("Starting {} experiments via {} backend".format(len(cfgs), backend))
        
        if backend == "ray":
            ray.init(address=basecfg.get("address", "auto"), _redis_password=basecfg.get("redis_password", None))
        
        if backend == "ray":
            configurations = [ray_eval_fit.options( 
                        num_cpus=basecfg.get("num_cpus", 1),
                        num_gpus=basecfg.get("num_gpus", 0),
                        memory = basecfg.get("max_memory", 1000 * 1024 * 1024) # 1 GB
                    ).remote(
                        basecfg.get("pre",None),
                        basecfg.get("fit", None),
                        basecfg.get("post",None),
                        os.path.join(basecfg["out_path"], str(experiment_id)),
                        experiment_id,
                        cfg
                    ) for experiment_id, cfg in enumerate(cfgs)
            ]
            print("SUBMITTED JOBS, NOW WAITING")
        else:
            configurations = [
                    (
                    basecfg.get("pre",None),
                    basecfg.get("fit", None),
                    basecfg.get("post",None),
                    os.path.join(basecfg["out_path"], str(experiment_id)),
                    experiment_id,
                    cfg
                ) for experiment_id, cfg in enumerate(cfgs)
            ]

        if backend == "ray":
            # https://github.com/ray-project/ray/issues/8164
            def to_iterator(configs):
                while configs:
                    result, configs = ray.wait(configs)
                    yield ray.get(result[0])

            random.shuffle(configurations)
            for result in tqdm(to_iterator(configurations), total=len(configurations)):
                experiment_id, results, out_file_content = result 
                with open(basecfg["out_path"] + "/results.jsonl", "a", 1) as out_file:
                    out_file.write(out_file_content)

        elif backend == "multiprocessing":
            pool = Pool(basecfg.get("num_cpus", 1))
            for eval_return in tqdm(pool.imap_unordered(eval_fit, configurations), total = len(configurations), disable = not verbose):
                experiment_id, results, out_file_content = eval_return
                with open(basecfg["out_path"] + "/results.jsonl", "a", 1) as out_file:
                    out_file.write(out_file_content)
        else:
            for f in tqdm(configurations, disable = not verbose):
                eval_return = eval_fit(f)
                experiment_id, results, out_file_content = eval_return
                with open(basecfg["out_path"] + "/results.jsonl", "a", 1) as out_file:
                    out_file.write(out_file_content)
    except Exception as e:
        return_str = str(e) + "\n"
        return_str += traceback.format_exc() + "\n"
    finally:
        print(return_str)
        if backend == "ray":
            ray.shutdown()