import plotly.express as px
import pandas as pd
import plotly.io as pio
import os, json
import numpy as np

polyfem_exe = "/home/zizhou/polyfem/build/release/PolyFEM_bin"

visc_list = [1, 0.1, 0.01]
ref_list = [0, 1, 2]
orders = [[1, 1], [2, 1]]

with open('data.csv','w') as file:
    file.write("visc, ref, v_order, p_order, err\n")
    for visc in visc_list:
        for ref in ref_list:
            for order_pair in orders:
                with open("/home/zizhou/polyfem/run.json") as f:
                    data = json.load(f)
                
                data["n_refs"] = ref
                data["time_steps"] *= 2**ref
                data["discr_order"] = order_pair[0]
                data["pressure_discr_order"] = order_pair[1]
                data["params"]["viscosity"] = visc
                data["problem_params"]["viscosity"] = visc

                with open("/home/zizhou/polyfem/tmp.json", 'w') as json_file:
                    json.dump(data, json_file, indent = 4)
                
                os.system(polyfem_exe + " --cmd -j /home/zizhou/polyfem/tmp.json")
                with open('runtime.csv','r') as f:
                    lines = f.readlines()
                    err = float(lines[0][:-1])
                
                file.write(",".join(map(str, [visc, ref, order_pair[0], order_pair[1], err]))+"\n")
