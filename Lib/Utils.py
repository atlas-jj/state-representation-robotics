import numpy as np
import torch
import torch.utils.data as Data
import os, fnmatch, sys
from PIL import Image, ImageDraw, ImageChops, ImageFilter
from torchvision import datasets, models, transforms

# <editor-fold desc="previous code, not useful now">
def convert2_region_domain(trajs, n_region):
    new_trajs = []
    num_trajs = len(trajs)
    for i in range(num_trajs):
        c = trajs[i]
        if c.shape[0] <= n_region:
            new_trajs.append(c)
        else:
            num_states = c.shape[0] - n_region + 1
            d = np.zeros((num_states, n_region * c.shape[1]))
            for j in range(num_states):
                d[j, :] = c[j:j+n_region,:].flatten()
            new_trajs.append(d)
    return new_trajs


def normalize_trajs(trajs):
    maxs = []
    mins = []
    num_trajs = len(trajs)
    for i in range(num_trajs):
        maxs.append(trajs[i].max())
        mins.append(trajs[i].min())
    amin = min(mins)
    amax = max(maxs)
    for i in range(num_trajs):
        trajs[i] = (trajs[i] - amin)/(amax - amin)
    return amax, amin, trajs


def prepare_dataset(trajs, train_test_mode=0):
    """
    Stack all trajectories, a dataset should have only one target
    :param
    trajs_file_path:
    train_test_mode: 0: all for training / testing. 1: split training / testing | default 0
    :return:
    all_stacked_states
    all_stacked_actions
    target_state (average)
    """
    # amax, amin, trajs = normalize_trajs(trajs)
    num_trajs = len(trajs)
    states = []
    actions = []
    targets = []

    test_states = []
    test_actions = []
    if train_test_mode is 1:
        split = 0.6  # split*100 % for training, and the remaining for testing
    else:
        split = 1

    for i in range(num_trajs):
        c = trajs[i]
        # append the target
        targets.append(c[c.shape[0] - 1, :])
        if i < (num_trajs * split):  # for training
            append_states_actions(c, states, actions)
        else:
            append_states_actions(c, test_states, test_actions)

    # list objects to np array
    # states: num_all_states * state_dim
    # actions: same dim as states
    if train_test_mode is 0:
        return np.array(states), np.array(actions), np.average((np.array(targets)), axis=0)
    else:
        return np.array(states), np.array(actions), np.array(test_states), np.array(test_actions), \
               np.average((np.array(targets)), axis=0)


def append_states_actions(c, p_states, p_actions):
    for j in range(c.shape[0] - 1):
        p_actions.append(c[j + 1, :] - c[j, :])
        p_states.append(c[j, :])

    # append last state
    p_states.append(c[c.shape[0] - 1, :])
    # last row action for last state are zeros
    p_actions.append(np.zeros(c.shape[1]))


def wrap_data_loader(trajs_file_path, kwargs, train_test_mode=0, batch_size=128, shuffle=True):
    """
    return a dataloader and the target state (np array)
    :param trajs_file_path:
    :param kwargs:
    :param train_test_mode:
    :param batch_size:
    :param shuffle:
    :return:
    """
    trajs = np.load(trajs_file_path)

    if train_test_mode is 0:
        states, actions, target = prepare_dataset(trajs, 0)
        dataset = Data.TensorDataset(torch.from_numpy(states), torch.from_numpy(actions))
        mydataLoader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return mydataLoader, target
    else:
        states, actions, test_states, test_actions, target = prepare_dataset(trajs, 1)
        dataset = Data.TensorDataset(torch.from_numpy(states), torch.from_numpy(actions))
        trainDataLoader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_dataset = Data.TensorDataset(torch.from_numpy(test_states), torch.from_numpy(test_actions))
        testDataLoader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return trainDataLoader, testDataLoader, target


def wrap_data_loader_regions(trajs_file_path, kwargs, train_test_mode=0, batch_size=128, shuffle=True, n_region=10):
    """
    return a dataloader and the target state (np array)
    :param trajs_file_path:
    :param kwargs:
    :param train_test_mode:
    :param batch_size:
    :param shuffle:
    :return:
    """
    trajs = np.load(trajs_file_path)
    trajs_new = convert2_region_domain(trajs, n_region)
    if train_test_mode is 0:
        states, actions, target = prepare_dataset(trajs_new, 0)
        dataset = Data.TensorDataset(torch.from_numpy(states), torch.from_numpy(actions))
        mydataLoader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return mydataLoader, target
    else:
        states, actions, test_states, test_actions, target = prepare_dataset(trajs_new, 1)
        dataset = Data.TensorDataset(torch.from_numpy(states), torch.from_numpy(actions))
        trainDataLoader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_dataset = Data.TensorDataset(torch.from_numpy(test_states), torch.from_numpy(test_actions))
        testDataLoader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return trainDataLoader, testDataLoader, target
# </editor-fold>


def estimate_normalDist_variance(conf_level):
    # return 1
    if conf_level >= 0.99999943:
        return 0.2*0.2
    elif conf_level >= 0.999:
        return 0.3*0.3
    elif conf_level >= 0.988:
        return 0.4*0.4
    elif conf_level >= 0.974:
        return 0.45*0.45
    elif conf_level >= 0.955:
        return 0.5*0.5
    elif conf_level >= 0.931:
        return 0.55*0.55
    elif conf_level >= 0.904:
        return 0.6*0.6
    elif conf_level >= 0.876:
        return 0.65*0.65
    elif conf_level >= 0.847:
        return 0.7*0.7
    elif conf_level >= 0.82:
        return 0.75*0.75
    else:
        return 0.5*0.5

def get_files(image_folder, pattern):
    listOfFiles = os.listdir(image_folder)
    img_names = []
    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, pattern):
            img_names.append(entry)
    return img_names


def load_images(image_folder, img_size):
    image_names = get_files(image_folder, "*.jpg")
    # load images and transform, and convert to pytorch object
    transform = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize((0.5,), (0.5,)), #  image = (image - mean) / std， convert to -1,1
        ]
    )
    samples = torch.zeros((len(image_names), 3, img_size, img_size))
    for i in range(len(image_names)):
        img = Image.open(image_folder + '/' + image_names[i])
        img = img.resize((img_size, img_size))
        samples[i] = transform(img)
    return samples

def load_images_sequence(image_folder, img_num, img_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,)), #  image = (image - mean) / std， convert to -1,1
    ]
    )
    samples = torch.zeros((img_num, 3, img_size, img_size))
    for i in range(img_num):
        img = Image.open(image_folder + '/raw_' + str(i+1) + '.jpg')
        img = img.resize((img_size, img_size))
        samples[i] = transform(img)
    return samples


def get_image_tensor(img_path, img_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,)), #  image = (image - mean) / std， convert to -1,1
    ]
    )
    sample = torch.zeros((1, 3, img_size, img_size))

    img = Image.open(img_path)
    img = img.resize((img_size, img_size))
    sample[0] = transform(img)
    return sample

def get_image_tensor_from_pil(pil_img, img_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,)), #  image = (image - mean) / std， convert to -1,1
    ]
    )
    sample = torch.zeros((1, 3, img_size, img_size))
    img = pil_img.resize((img_size, img_size))
    sample[0] = transform(img)
    return sample

def wrap_data_loader_images(image_folder, img_size, kwargs, train_test_mode=0, batch_size=128, shuffle=True):
    """
    return a dataloader and the target state (np array)
    :param trajs_file_path:
    :param kwargs:
    :param train_test_mode:
    :param batch_size:
    :param shuffle:
    :return:
    """
    samples = load_images(image_folder, img_size)

    if train_test_mode is 1:
        split = 0.7  # split*100 % for training, and the remaining for testing
    else:
        split = 1

    train_max_it = int(samples.shape[0] * split)

    if train_test_mode is 0:
        dataset = Data.TensorDataset(samples)
        mydataLoader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return mydataLoader
    else:
        dataset = Data.TensorDataset(samples[0:train_max_it, :])
        trainDataLoader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_dataset = Data.TensorDataset(samples[train_max_it:samples.shape[0], :])
        testDataLoader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return trainDataLoader, testDataLoader

def clear_all_files(folder):
    # if does not exist, create one
    if not os.path.exists(folder):
        os.makedirs(folder)

    image_names = get_files(folder, "*.*")
    for i in range(len(image_names)):
        os.remove('./' + folder + '/' + image_names[i])

def d2s(d, n_digits=4):
    strf = "{0:." + str(n_digits) + "f}"
    return str(float((strf.format(d))))

def norm2range(x, beta):
    """
    normalize to [0,1], origin[-beta, beta]
    :param x:
    :param beta:
    :return:
    """
    return x/(2*beta) + 0.5