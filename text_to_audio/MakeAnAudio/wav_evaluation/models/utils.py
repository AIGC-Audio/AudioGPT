import argparse
import yaml
import sys

def read_config_as_args(config_path,args=None,is_config_str=False):
    return_dict = {}

    if config_path is not None:
        if is_config_str:
            yml_config = yaml.load(config_path, Loader=yaml.FullLoader)
        else:
            with open(config_path, "r") as f:
                yml_config = yaml.load(f, Loader=yaml.FullLoader)

        if args != None:
            for k, v in yml_config.items():
                if k in args.__dict__:
                    args.__dict__[k] = v
                else:
                    sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))
        else:
            for k, v in yml_config.items():
                return_dict[k] = v

    args = args if args != None else return_dict
    return argparse.Namespace(**args)
