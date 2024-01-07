import configs
import tasks

prog_args = configs.arg_parse()

if prog_args.dataset is not None:
    if prog_args.dataset == "mnist":
        print("Explain mnist dataset")
        explaining_task = "tasks.mnist"
        eval(explaining_task)(prog_args)
    
    elif prog_args.dataset == "fashion":
        print("Explain fashion mnist dataset")
        explaining_task = "tasks.fashion"
        eval(explaining_task)(prog_args)
                
    else:
        print("Unknown dataset")