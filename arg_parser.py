import argparse

# global variables
MIN = 0.4
MAX = 0.8

def limited_ratio_type(arg):
    '''
    Type function for argparse ratio
    
    :param arg: value for ratio
    :returns: float within MIN and MAX
    '''

    try:
        f = float(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < MIN or f > MAX:
        raise argparse.ArgumentTypeError("Argument must be < " + str(MAX) + " and > " + str(MIN))
    return f


def parse_args():
    '''
    builds a parser for 
    command line arguments

    :returns: args values
    '''

    parser = argparse.ArgumentParser()

    # options
    parser.add_argument("--seed", "-sd", default=177, type=int, help="choose the seed for rng functions in the code")
    parser.add_argument("--dataset", "-dt", default="pepper", choices=["pepper", "swat", "voraus"], type=str, help="choose which dataset to load")
    parser.add_argument("--ratio", "-rt", default=0.8, type=limited_ratio_type, help=f"a ratio between {MIN} and {MAX} for deciding how large the training set will be")
    parser.add_argument("--normalize", "-n", action="store_true", help="if set, data is normalized")
    parser.add_argument("--downsample", "-ds", action="store_true", help="if set, data is downsampled(only for SWaT dataset)")
    parser.add_argument("--plot", "-plt", action= "store_true", help="if set, metric plots are shown(only for model evaluation)")

    return parser.parse_args()