import configs
import tasks

prog_args = configs.arg_parse()

if prog_args.experiment is not None:
    if prog_args.experiment == "synthetic":
        print("Synthetic experiment")
        explaining_task = "tasks.synthetic"
        eval(explaining_task)(prog_args) 
    else:
        print("Unknown experiments")