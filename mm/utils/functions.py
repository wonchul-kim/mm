import yaml

def add_params_to_args(args, params_file):
    with open(params_file) as yf:
        params = yaml.load(yf, Loader=yaml.FullLoader)

    for key, value in params.items():
        setattr(args, key, value)