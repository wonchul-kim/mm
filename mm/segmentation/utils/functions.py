import yaml

def add_params_to_args(args, params_file):
    with open(params_file) as yf:
        params = yaml.load(yf, Loader=yaml.FullLoader)

    for key, value in params.items():
        setattr(args, key, value)
        

def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if cfg.show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = cfg.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg