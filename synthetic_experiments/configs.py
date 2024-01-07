import argparse
    
def arg_parse():
    parser = argparse.ArgumentParser(description="EMaP arguments.")

    parser.add_argument("--experiment", dest="experiment", help="Experiment.")
    
    parser.add_argument("--syn-sub", dest="syn_sub", help="Shape of synthetic experiments.")
    
    
    parser.add_argument(
            "--no-points", dest="no_points", type=int, help="Number of data points."
    )
    
    parser.add_argument(
            "--no-runs", dest="no_runs", type=int, help="Number of runs."
    )
    
    parser.add_argument(
            "--data-noise", dest="data_noise", type=float, help="Amount of noise on the data."
    )
    
    parser.add_argument(
            "--sampler-noise", dest="sampler_noise", type=float, help="Amount of noise to learn the low-dim hyperplanes."
    )
    
    parser.set_defaults(
            experiment="synthetic",
            syn_sub="circle",
            no_points = 400,
            data_noise = 0.01,
            no_runs = 1,
            sampler_noise = 1.0
        )
    
    return parser.parse_args()

def arg_parse_for_notebook():
    parser = argparse.ArgumentParser(description="EMaP arguments.")

    parser.add_argument("--experiment", dest="experiment", help="Experiment.")
    
    parser.add_argument("--syn-sub", dest="syn_sub", help="Shape of synthetic experiments.")
    
    
    parser.add_argument(
            "--no-points", dest="no_points", type=int, help="Number of data points."
    )
    
    parser.add_argument(
            "--data-noise", dest="data_noise", type=float, help="Amount of noise on the data."
    )
    
    parser.add_argument(
            "--sampler-noise", dest="sampler_noise", type=float, help="Amount of noise to learn the low-dim hyperplanes."
    )
    
    parser.set_defaults(
            experiment="synthetic",
            syn_sub="circle",
            no_points = 400,
            data_noise = 0.01,
            sampler_noise = 1.0
        )
    
    return parser