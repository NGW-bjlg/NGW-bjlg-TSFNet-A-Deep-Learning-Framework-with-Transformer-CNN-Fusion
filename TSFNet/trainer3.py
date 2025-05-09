import logging
import os
import random
import sys
import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
import utils
import rasterio
from rasterio.windows import Window
from rasterio.errors import RasterioError
from torch.utils.data.dataset import IterableDataset
from collections import Counter


import math
def data_pretrain(img):
    img = np.nan_to_num(img, nan=-1, copy=True)
    replacement_value = -1
    condition1 = abs(img) > 100
    img[condition1] = replacement_value
    B_red = img[:, :, 0]
    B_green = img[:, :, 1]
    B_blue = img[:, :, 2]
    B_nir = img[:, :, 3]
    B_mir = img[:, :, 4]
    B_swir = img[:, :, 5]
    NDVI = (B_nir- B_red)/(B_nir+B_red)
    NDBI = (B_swir- B_nir)/(B_swir+ B_nir)
    MNDWI = (B_green- B_mir)/(B_green+ B_mir)
    para_r_0 = (3*abs(B_nir-B_red))
    para_r_0[para_r_0==0]=1
    para_r = 1-(B_mir-B_nir)/(para_r_0)
    para_v = B_red-B_green
    condition2 = para_r * para_v >=0
    condition3 = para_r * para_v < 0
    a = -1 * abs((B_mir - B_blue)/(B_mir + B_blue))
    b = (B_mir - B_blue) / (B_mir + B_blue)
    a[condition3] = 0
    b[condition2] = 0
    NDBSI = a+b
    img = np.concatenate((img, NDVI[:, : ,np.newaxis],NDBI[:, : ,np.newaxis],MNDWI[:, : ,np.newaxis],NDBSI[:, : ,np.newaxis]), axis=2)
    return img

def relabel(label1):
    label = label1.copy()
    label[label==4] = 3
    label[label==5] =4
    label[label == 6] = 4
    label[label ==7]=5
    label[label == 9] = 5
    label[label ==8]=6
    mask1=  label<1
    mask2= label>6
    label[mask1+mask2]=0
    return label

def label_pretrain(dw_label, ld_label,ms_label,dw1_label):
    label = relabel(ld_label)
    ld_label = relabel(ld_label)
    dw_label = relabel(dw_label)
    mask1 = dw_label != ld_label
    label[mask1] = 7
    label[dw_label == 6] = 6
    label[ms_label == 3] = 3
    label[ms_label == 5] = 4
    label[ms_label == 6] = 4

    dw_label[ms_label == 3] = 3
    dw_label[ms_label == 5] = 4
    dw_label[ms_label == 6] = 4

    ld_label[dw_label == 6] = 6
    ld_label[ms_label == 3] = 3
    ld_label[ms_label == 5] = 4
    ld_label[ms_label == 6] = 4
    return label, dw_label,ld_label

class StreamingGeospatialDataset(IterableDataset):
    
    def __init__(self, imagery_fns, dw_label_fns=None, ld_label_fns=None, pre_label_fns=None,groups=None, chip_size=256, num_chips_per_tile=200, windowed_sampling=False, image_transform=None, label_transform=None, nodata_check=None, verbose=False):
        if dw_label_fns is None:
            self.fns = imagery_fns
            self.use_labels = False
        else:
            self.fns = list(zip(imagery_fns, dw_label_fns, ld_label_fns, pre_label_fns))
            
            self.use_labels = True

        self.groups = groups

        self.chip_size = chip_size
        self.num_chips_per_tile = num_chips_per_tile
        self.windowed_sampling = windowed_sampling

        self.image_transform = image_transform
        self.label_transform = label_transform
        self.nodata_check = nodata_check

        self.verbose = verbose

        if self.verbose:
            print("Constructed StreamingGeospatialDataset")

    def stream_tile_fns(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None: # In this case we are not loading through a DataLoader with multiple workers
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        # We only want to shuffle the order we traverse the files if we are the first worker (else, every worker will shuffle the files...)
        if worker_id == 0:
            np.random.shuffle(self.fns) # in place

        if self.verbose:
            print("Creating a filename stream for worker %d" % (worker_id))

        # This logic splits up the list of filenames into `num_workers` chunks. Each worker will recieve ceil(num_filenames / num_workers) filenames to generate chips from. If the number of workers doesn't divide the number of filenames evenly then the last worker will have fewer filenames.
        N = len(self.fns)
        num_files_per_worker = int(np.ceil(N / num_workers))
        lower_idx = worker_id * num_files_per_worker
        upper_idx = min(N, (worker_id+1) * num_files_per_worker)
        for idx in range(lower_idx, upper_idx):

            label_fn = None
            if self.use_labels:
                
                img_fn, dw_label_fn, ld_label_fn,pre_label_fn = self.fns[idx]
            else:
                img_fn = self.fns[idx]

            if self.groups is not None:
                group = self.groups[idx]
            else:
                group = None

            if self.verbose:
                print("Worker %d, yielding file %d" % (worker_id, idx))

            yield (img_fn, dw_label_fn, ld_label_fn, pre_label_fn,group)

    def stream_chips(self):
        height_all = np.ones((89,6))
        aaa=0
        for img_fn, dw_label_fn, ld_label_fn,pre_label_fn, group in self.stream_tile_fns():
            num_skipped_chips = 0

            # Open file pointers
            img_fp = rasterio.open(img_fn, "r")
            dw_label_fp = rasterio.open(dw_label_fn, "r") if self.use_labels else None
            ld_label_fp = rasterio.open(ld_label_fn, "r") if self.use_labels else None
            pre_label_fp = rasterio.open(pre_label_fn, "r") if self.use_labels else None
            # dw1_label_fp = rasterio.open(dw1_label_fn, "r") if self.use_labels else None
            height, width = img_fp.shape
            if self.use_labels: # garuntee that our label mask has the same dimensions as our imagery
                t_height, t_width = dw_label_fp.shape
                t_height1, t_width1 = ld_label_fp.shape
                height_all[aaa,:] = [height,t_height,t_height1,width,t_width,t_width1]
                aaa+=1
                if t_height != height:
                    print('error')
                assert height == t_height and width == t_width
                assert height == t_height1 and width == t_width1



            # If we aren't in windowed sampling mode then we should read the entire tile up front
            img_data = None
            label_data = None
            try:
                if not self.windowed_sampling:
                # if self.windowed_sampling:
                    img_data = np.rollaxis(img_fp.read(window=Window(0, 0, t_height, t_width)), 0, 3)
                    img_data = data_pretrain(img_data)
                    dw_labels = dw_label_fp.read(window=Window(0, 0, t_height, t_width)).squeeze()
                    ld_labels = ld_label_fp.read(window=Window(0, 0, t_height, t_width)).squeeze()
                    dw1_labels = dw1_label_fp.read(window=Window(0, 0, t_height, t_width)).squeeze()
                    label = label_pretrain(dw_labels.astype(int),ld_labels,dw1_labels.astype(int))
                    if self.use_labels:
                        label_data = label_fp.read().squeeze() # assume the label geotiff has a single channel
            except RasterioError as e:
                print("WARNING: Error reading in entire file, skipping to the next file")
                continue
            # chip_number_x = [0,287,0,287,144]
            # chip_number_y =[0,0,287,287,144]
            chip_coordinates = []  # upper left coordinate (y,x), of each chip that this Dataset will return
            # a= list(range(0, height - self.chip_size, stride)) + [height - self.chip_size]
            # image_all = np.rollaxis(img_fp.read(), 0, 3)
            for y in list(range(0, height - self.chip_size, self.chip_size)) + [height - self.chip_size]:
                for x in list(range(0, width - self.chip_size,self.chip_size)) + [width - self.chip_size]:
                    chip_coordinates.append((x, y))
            num_chips = len(chip_coordinates)
            for i in range(num_chips):
                # Select the top left pixel of our chip randomly
                x,y = chip_coordinates[i]
                # Read imagery / labels
                img = None
                labels = None
                if self.windowed_sampling:
                    try:
                        img = np.rollaxis(img_fp.read(window=Window(x, y, self.chip_size, self.chip_size)), 0, 3)
                        img = data_pretrain(img)
                        if img[223,223,0]==0:
                            continue
                        # print(img.shape)
                        if self.use_labels:
                            dw_labels = dw_label_fp.read(window=Window(x, y, self.chip_size, self.chip_size)).squeeze()
                            ld_labels = ld_label_fp.read(window=Window(x, y, self.chip_size, self.chip_size)).squeeze()
                            labels = pre_label_fp.read(window=Window(x, y, self.chip_size, self.chip_size)).squeeze()
                            # dw1_labels = dw1_label_fp.read(window=Window(x, y, self.chip_size, self.chip_size)).squeeze()
                            # labels, dw_labels, ld_labels= label_pretrain(dw_labels.astype(int), ld_labels, ms_labels,dw1_labels.astype(int))

                            # labels = label_fp.read(window=Window(x, y, self.chip_size, self.chip_size)).squeeze()
                    except RasterioError:
                        print("WARNING: Error reading chip from file, skipping to the next chip")
                        continue
                else:
                    img = img_data[y:y+self.chip_size, x:x+self.chip_size, :]
                    if self.use_labels:
                        labels = label_data[y:y+self.chip_size, x:x+self.chip_size]

                # Check for no data
                if self.nodata_check is not None:
                    if self.use_labels:
                        skip_chip = self.nodata_check(img, labels)
                    else:
                        skip_chip = self.nodata_check(img)

                    if skip_chip: # The current chip has been identified as invalid by the `nodata_check(...)` method
                        num_skipped_chips += 1
                        continue

                # Transform the imagery
                if self.image_transform is not None:
                    if self.groups is None:
                        img = self.image_transform(img)
                        # img = img
                    else:
                        img = self.image_transform(img, group)
                else:
                    img = torch.from_numpy(img).squeeze()

                # Transform the labels
                if self.use_labels:
                    if self.label_transform is not None:
                        if self.groups is None:
                            labels = labels.astype(np.int64)
                            dw_labels = dw_labels.astype(np.int64)
                            ld_labels = ld_labels.astype(np.int64)
                            # labels = self.label_transform(labels)
                        else:
                            print(label_fn)
                            labels = self.label_transform(labels, group)
                            print(labels)
                    else:
                        labels = torch.from_numpy(labels).squeeze()

                # Note, that img should be a torch "Double" type (i.e. a np.float32) and labels should be a torch "Long" type (i.e. np.int64)
                if self.use_labels:
                     yield img, labels, dw_labels, ld_labels
                else:
                     yield img
            # Close file pointers
            img_fp.close()
            if self.use_labels:
                dw_label_fp.close()
                ld_label_fp.close()

            if num_skipped_chips>0 and self.verbose:
                print("We skipped %d chips on %s" % (img_fn))
        # testend = 1

    def __iter__(self):
        if self.verbose:
            print("Creating a new StreamingGeospatialDataset iterator")
        return iter(self.stream_chips())

def image_transforms(img):
    # img = (img - utils.IMAGE_MEANS) / utils.IMAGE_STDS
    img = np.rollaxis(img, 2, 0).astype(np.float32)
    img = torch.from_numpy(img)
    return img

def label_transforms(labels):
    labels = utils.LABEL_CLASS_TO_IDX_MAP[labels]
    labels = torch.from_numpy(labels)
    return labels
def nodata_check(img, labels):
    return False
    # return np.any(labels == 0) or np.any(np.sum(img == 0, axis=2) == 4)

def trainer_dataset(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    batch_size = args.batch_size

    #-------------------
    # Load input data
    #-------------------
    
    input_dataframe = pd.read_csv(args.list_dir)
    image_fns = input_dataframe["image_fn"].values
    dw_label_fns = input_dataframe["dw_label_fn"].values
    ld_label_fns = input_dataframe["ld_label_fn"].values
    pre_label_fns = input_dataframe["pre_label_fn"].values
    # dw1_label_fns = input_dataframe["dw1_label_fn"].values
    NUM_CHIPS_PER_TILE =300  # How many chips will be sampled from one large-scale tile
    CHIP_SIZE = 224 # Size of each sampled chip
    db_train = StreamingGeospatialDataset(
        imagery_fns=image_fns, dw_label_fns=dw_label_fns,ld_label_fns=ld_label_fns, pre_label_fns=pre_label_fns, groups=None, chip_size=CHIP_SIZE, num_chips_per_tile=NUM_CHIPS_PER_TILE, windowed_sampling=True, verbose=False,
        image_transform=image_transforms, label_transform=label_transforms,nodata_check=nodata_check
    ) #

    print("The length of train set is: {}".format(len(image_fns)*NUM_CHIPS_PER_TILE))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    model.train()
    ignore_index = 7
    ce_loss = CrossEntropyLoss(ignore_index=ignore_index)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    num_class=7
    max_epoch = args.max_epochs
    num_training_batches_per_epoch = int(len(image_fns) * NUM_CHIPS_PER_TILE / batch_size)
    max_iterations = args.max_epochs * len(image_fns)*NUM_CHIPS_PER_TILE
    logging.info("{} iterations per epoch. {} max iterations ".format(len(image_fns)*NUM_CHIPS_PER_TILE, max_iterations))
    iterator = range(max_epoch)
    for epoch_num in iterator:
        loss1 = []
        loss2 = []
        loss3 = []
        loss_all = []
        class_total = np.zeros(num_class)
        correct_predictions = np.zeros(num_class)
        if epoch_num ==0:
            class_weights = torch.ones(num_class)
            all_targets = []
        for i_batch, (image_batch,label_batch,dw_label_batch,ld_label_batch) in tqdm(enumerate(trainloader),  total=num_training_batches_per_epoch):
            image_batch, label_batch,dw_label_batch,ld_label_batch = image_batch.cuda(), label_batch.cuda(),dw_label_batch.cuda(),ld_label_batch.cuda()
            if epoch_num==0:
                all_targets.append(label_batch)

            outputs1,outputs2,outputs3 = model(image_batch)
            t_output = F.softmax((outputs2), dim=1) # Created mask label
            t_output = t_output.argmax(axis=1)

            loss_ce1 = ce_loss(outputs1, dw_label_batch[:].long()) # General CE loss for CNN branch
            loss_ce2 = ce_loss(outputs2, label_batch[:].long()) # Mask CE (mce) loss for ViT branch
            loss_ce3 = ce_loss(outputs3, ld_label_batch[:].long())

            loss = loss_ce2
            optimizer.zero_grad()
            for class_idx in range(num_class):
                class_mask = label_batch == class_idx
                class_total[class_idx] += class_mask.sum().item()

                if class_total[class_idx] > 0:
                    correct_predictions[class_idx] += (t_output[class_mask] == label_batch[class_mask]).sum()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=3.0)
            optimizer.step()
            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            lr_ = base_lr * 0.3 ** (epoch_num // 10)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            loss1.append(loss_ce1.item())
            loss2.append(loss_ce2.item())
            loss3.append(loss_ce3.item())
            loss_all.append(loss.item())
            iter_num = iter_num + 1
        producer_accuracy = correct_predictions / class_total

        print("producer_accuracy:", producer_accuracy)
        if epoch_num <0:
            all_targets = torch.cat(all_targets, dim=0)
            class_counts = Counter(all_targets.view(-1).tolist())
            for c in range(num_class):
                class_weights[c] = 1.0 / (class_counts.get(c, 1) + 1e-5)  # 类别越少，权重越高

            # 归一化权重
            class_weights[0] = class_weights[0]/5
            class_weights = class_weights / class_weights.sum()
            print(class_counts)
            print(class_weights)
            # class_weights = np.array[1,1,1,1,5,1,1,1]
            ce_loss = CrossEntropyLoss(ignore_index=ignore_index,weight=class_weights.cuda())

        if epoch_num <0:
            class_weights = torch.ones(num_class)
            class_weights[0] = 0.05
            class_weights[4] = 2
            class_weights[5] = 2
            class_weights[6] = 3
            ce_loss = CrossEntropyLoss(ignore_index=ignore_index)
        #     ce_loss = CrossEntropyLoss(ignore_index=10)
        avg_loss1 = np.mean(loss1)
        avg_loss2=np.mean(loss2)
        avg_loss3 = np.mean(loss3)
        avg_loss = np.mean(loss_all)
        logging.info('Epoch : %d, CE-branch1 : %f, MCE-branch2: %f, CE-branch3 : %f,loss: %f' % (epoch_num, avg_loss1, avg_loss2, avg_loss3,avg_loss))
        save_interval = 10
        if epoch_num  % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            # iterator.close()
            break

    writer.close()
    return "Training Finished!"