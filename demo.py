import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from network import RetinaNet
from detection_utils import compute_targets, get_detections, set_seed
from predict import validate, test
from tensorboardX import SummaryWriter
from absl import app, flags
import numpy as np
from dataset import CocoDataset, Resizer, Normalizer, collater
from torchvision import transforms
import losses
import logging
import time

# Import necessary modules for gradient clipping and data augmentation
from torch.nn.utils import clip_grad_norm_
import random

FLAGS = flags.FLAGS
flags.DEFINE_float('lr', 1e-2, 'Learning Rate')  # Increased learning rate
flags.DEFINE_float('momentum', 0.9, 'Momentum for optimizer')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight Decay for optimizer')
flags.DEFINE_string('output_dir', 'runs/run1/', 'Output Directory')
flags.DEFINE_integer('batch_size', 2, 'Batch Size')  # Increased batch size
flags.DEFINE_integer('seed', 2, 'Random seed')
flags.DEFINE_integer('max_iter', 100000, 'Total Iterations')
flags.DEFINE_integer('val_every', 10000, 'Iterations interval to validate')
flags.DEFINE_integer('save_every', 50000, 'Iterations interval to save model')
flags.DEFINE_integer('preload_images', 1,
    'Whether to preload train and val images at beginning of training.')
flags.DEFINE_multi_integer('lr_step', [60000, 80000], 'Iterations to reduce learning rate')

log_every = 20

def setup_logging():
    log_formatter = logging.Formatter(
        '%(asctime)s: %(levelname)s %(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logging.getLogger().handlers = []
    if not len(logging.getLogger().handlers): 
        logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(logging.INFO)

def logger(tag, value, global_step):
    if tag == '':
       logging.info('')
    else:
       logging.info(f'  {tag:>15s} [{global_step:07d}]: {value:5f}')

# Define RandomHorizontalFlip transform that adjusts bounding boxes
class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        if random.random() < 0.5:
            image = image[:, ::-1, :]
            _, width, _ = image.shape

            x_min = annots[:, 0].copy()
            x_max = annots[:, 2].copy()
            annots[:, 0] = width - x_max
            annots[:, 2] = width - x_min
            sample = {'img': image, 'annot': annots}

        return sample

def main(_):
    setup_logging()
    torch.set_num_threads(4)
    torch.manual_seed(FLAGS.seed)
    set_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Update transforms to include RandomHorizontalFlip
    transform_train = transforms.Compose([Normalizer(), RandomHorizontalFlip(), Resizer()])
    transform_val = transforms.Compose([Normalizer(), Resizer()])

    dataset_train = CocoDataset('train', seed=FLAGS.seed,
        preload_images=FLAGS.preload_images > 0,
        transform=transform_train)
    dataset_val = CocoDataset('val', seed=0, 
        preload_images=FLAGS.preload_images > 0,
        transform=transform_val)
    dataloader_train = DataLoader(dataset_train, batch_size=FLAGS.batch_size,
                                  num_workers=3, collate_fn=collater, pin_memory=True)

    model = RetinaNet(p67=True, fpn=True)

    num_classes = dataset_train.num_classes
    device = torch.device('cuda:0')
    # For Mac users
    # device = torch.device("mps") 
    model.to(device)

    writer = SummaryWriter(FLAGS.output_dir, max_queue=1000, flush_secs=120)
    optimizer = torch.optim.SGD(model.parameters(), lr=FLAGS.lr, 
                                momentum=FLAGS.momentum, 
                                weight_decay=FLAGS.weight_decay)

    # Remove the existing scheduler
    # milestones = [int(x) for x in FLAGS.lr_step]
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=milestones, gamma=0.1)

    optimizer.zero_grad()
    dataloader_iter = None

    times_np, cls_loss_np, bbox_loss_np, total_loss_np = [], [], [], []

    # Define warmup iterations
    warmup_iters = 2000
    milestones = [int(x) for x in FLAGS.lr_step]

    # Use Focal Loss for classification
    lossFunc = losses.LossFunc(use_focal_loss=True)  # Make sure your LossFunc accepts this parameter

    max_norm = 0.1  # For gradient clipping

    model.train()
    for i in range(FLAGS.max_iter):
        iter_start_time = time.time()

        if dataloader_iter is None or i % len(dataloader_iter) == 0:
            dataloader_iter = iter(dataloader_train)

        try:
            data = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader_train)
            data = next(dataloader_iter)

        image, cls, bbox, is_crowd, image_id, _ = data

        if len(bbox) == 0:
            continue

        image = image.to(device, non_blocking=True)
        bbox = bbox.to(device, non_blocking=True)
        cls = cls.to(device, non_blocking=True)

        outs = model(image)
        pred_clss, pred_bboxes, anchors = get_detections(outs)
        gt_clss, gt_bboxes = compute_targets(anchors, cls, bbox)

        pred_clss = pred_clss.sigmoid()

        classification_loss, regression_loss = lossFunc(pred_clss, pred_bboxes,
                                                        anchors, gt_clss,
                                                        gt_bboxes)
        cls_loss = classification_loss.mean()
        bbox_loss = regression_loss.mean()
        total_loss = cls_loss + bbox_loss

        if np.isnan(total_loss.item()):
            logging.error(f'Loss went to NaN at iteration {i+1}')
            break

        if np.isinf(total_loss.item()):
            logging.error(f'Loss went to Inf at iteration {i+1}')
            break

        optimizer.zero_grad()
        total_loss.backward()

        # Apply gradient clipping
        clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        # Adjust learning rate for warmup and milestones
        if i < warmup_iters:
            lr = FLAGS.lr * float(i + 1) / float(warmup_iters)
        elif i >= milestones[0] and i < milestones[1]:
            lr = FLAGS.lr * 0.1
        elif i >= milestones[1]:
            lr = FLAGS.lr * 0.01
        else:
            lr = FLAGS.lr

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # No need to call scheduler.step()
        # scheduler.step()

        # Some logging
        times_np.append(time.time() - iter_start_time)
        total_loss_np.append(total_loss.item())
        cls_loss_np.append(cls_loss.item())
        bbox_loss_np.append(bbox_loss.item())

        if (i+1) % log_every == 0:
            avg_time = np.mean(times_np)
            iteration_rate = len(times_np) / np.sum(times_np)
            writer.add_scalar('iteration_rate', iteration_rate, i+1)
            logger('iteration_rate', iteration_rate, i+1)
            writer.add_scalar('loss_box_reg', np.mean(bbox_loss_np), i+1)
            logger('loss_box_reg', np.mean(bbox_loss_np), i+1)
            writer.add_scalar('lr', lr, i+1)
            logger('lr', lr, i+1)
            writer.add_scalar('loss_cls', np.mean(cls_loss_np), i+1)
            logger('loss_cls', np.mean(cls_loss_np), i+1)
            writer.add_scalar('total_loss', np.mean(total_loss_np), i+1)
            logger('total_loss', np.mean(total_loss_np), i+1)
            times_np, cls_loss_np, bbox_loss_np, total_loss_np = [], [], [], []

        if (i+1) % FLAGS.save_every == 0:
            torch.save(model.state_dict(), f'{FLAGS.output_dir}/model_{i+1}.pth')

        if (i+1) % FLAGS.val_every == 0 or (i+1) == FLAGS.max_iter:
            logging.info(f'Validating at {i+1} iterations.')
            val_dataloader = DataLoader(dataset_val, num_workers=3, collate_fn=collater)
            result_file_name = f'{FLAGS.output_dir}/results_{i+1}_val.json'
            model.eval()
            validate(dataset_val, val_dataloader, device, model, result_file_name, writer, i+1)
            model.train()

    torch.save(model.state_dict(), f'{FLAGS.output_dir}/model_final.pth')

    # Save prediction result on test set
    dataset_test = CocoDataset('test', preload_images=False,
                               transform=transforms.Compose([Normalizer(), Resizer()])) 
    test_dataloader = DataLoader(dataset_test, num_workers=1, collate_fn=collater)
    result_file_name = f'{FLAGS.output_dir}/results_{FLAGS.max_iter}_test.json'
    model.eval()
    test(dataset_test, test_dataloader, device, model, result_file_name)

if __name__ == '__main__':
    app.run(main)
