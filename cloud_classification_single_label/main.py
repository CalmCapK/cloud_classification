import argparse
import os
import random
from torch.backends import cudnn
import yaml
#import yamlloader

from data_loader import get_loader
from solver import Solver

def main(config):
    if not os.path.exists(config['model_path']):
        os.makedirs(config['model_path'])
    if not os.path.exists(config['train_checkpoint_file']):
        f = open(config['train_checkpoint_file'], 'w')
        f.close()
    if not os.path.exists(config['valid_checkpoint_file']):
        f = open(config['valid_checkpoint_file'], 'w')
        f.close()
    if not os.path.exists(config['test_ans_file']):
        f = open(config['test_ans_file'], 'w')
        f.close()

    print(config) 
    train_loader = get_loader(image_path=config['train_path'],
                            image_size=config['image_size'],
                            batch_size=config['batch_size'],
                            shuffle=True,
                            num_workers=config['num_workers'],
                            mode='train')
    valid_loader = get_loader(image_path=config['valid_path'],
                            image_size=config['image_size'],
                            batch_size=config['batch_size'],
                            shuffle=True,
                            num_workers=config['num_workers'],
                            mode='valid')
    test_loader = get_loader(image_path=config['test_path'],
                            image_size=config['image_size'],
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=config['num_workers'],
                            mode='test')

    solver = Solver(config, train_loader, valid_loader, test_loader)

    # Train and sample the images
    if config['mode'] == 'train':
        solver.train()
    elif config['mode'] == 'test':
        solver.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='./config.yaml')
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()
    
    with open(args.config_file) as f:
        config = yaml.load(f)
        #config = yaml.load(f, Loader=yamlloader.ordereddict.CLoader)
    config['mode'] = args.mode
    
    main(config)
