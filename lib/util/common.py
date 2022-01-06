import torch


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{torch.cuda.device_count()} cuda device available.')
    print(f'Using {device} device.')
    return device
