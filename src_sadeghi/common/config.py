download = False
data_folder = '../../data'
ckpt_folder = '../ckpts'

num_trains = {
    'mnist': 60000,
    'cifar_100': 50000,
    'celeba': 162770,
    'facescrub': 38387,
    'adult': 30162,
    'german': 700,
    'yaleb': 1096,
}

num_tests = {
    'mnist': 10000,
    'cifar_100': 10000,
    'celeba': 19867,
    'facescrub': 16797,
    'adult': 15060,
    'german': 300,
    'yaleb': 190,
}

input_sizes = {
    'mnist': (28, 28),
    'cifar_100': (3, 32, 32),
    'celeba': (3, 32, 32),  # (3, 218, 178),
    'facescrub': (3, 32, 32),
    'adult': 123,
    'german': 24,
    'yaleb': 504,
}

flat_input_sizes = {
    'mnist': 28*28,
    'cifar_100': 3*32*32,
    'celeba': 3*32*32,  # (3, 218, 178),
    'facescrub': 3*32*32,
    'adult': 123,
    'german': 24,
    'yaleb': 504,
}


num_channels = {
    'mnist': 1,
    'cifar_100': 3,
    'celeba': 3,
    'facescrub': 3,
    'adult': 0,
    'german': 0,
    'yaleb': 0,
}
