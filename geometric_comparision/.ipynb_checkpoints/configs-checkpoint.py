import argparse
    
def arg_parse():
    parser = argparse.ArgumentParser(description="EMaP arguments.")

    parser.add_argument("--dataset", dest="dataset", help="Input dataset.")
    
    parser.add_argument(
            "--perturb-method", dest="perturb_method", type=str, help="Perturbation methods: random, emap"
    )
    
    parser.add_argument(
            "--num-perturbs", dest="num_perturbs", type=int, help="Number of perturbed sample using to generate explanations."
    )
    
    parser.add_argument(
            "--method", dest="method", type=str, help="Explanation methods: lime, ime, shap"
        )

    # TODO: Check argument usage
    parser.set_defaults(
            dataset="mnist",
            perturb_method = "random",
            num_perturbs=1000,
            method="lime"
        )
    
    return parser.parse_args()