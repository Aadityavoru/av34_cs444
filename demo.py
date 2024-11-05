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
import losses as losses
import logging
import time
from torch.nn.utils import clip_grad_norm_  # Import gradient clipping utility

import numpy as np


class RandomFlip(object):
    """Randomly flip an image horizontally with a probability and adjust bounding boxes."""
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image, bboxes, cls, is_crowd, image_id = sample

        if np.random.rand() < self.flip_prob:
            # Flip the image horizontally
            image = np.fliplr(image).copy()

            # Flip bounding boxes
            width = image.shape[1]
            bboxes[:, [0, 2]] = width - bboxes[:, [2, 0]]  # Adjust the bounding box coordinates

        return image, bboxes, cls, is_crowd, image_id

FLAGS = flags.FLAGS
flags.DEFINE_float('lr', 1e-4, 'Learning Rate')
flags.DEFINE_float('momentum', 0.9, 'Momentum for optimizer')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight Decay for optimizer')
flags.DEFINE_string('output_dir', 'runs/retina-net-basic/', 'Output Directory')
flags.DEFINE_integer('batch_size', 1, 'Batch Size')
flags.DEFINE_integer('seed', 2, 'Random seed')
flags.DEFINE_integer('max_iter', 100000, 'Total Iterations')
flags.DEFINE_integer('val_every', 10000, 'Iterations interval to validate')
flags.DEFINE_integer('save_every', 50000, 'Iterations interval to save model')
flags.DEFINE_integer('preload_images', 1, 
    'Whether to preload train and val images at beginning of training. Preloading takes about 7 minutes on campus cluster but speeds up training by a lot. Set to 0 to disable.')
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

def main(_):
    setup_logging()
    torch.set_num_threads(4)
    torch.manual_seed(FLAGS.seed)
    set_seed(FLAGS.seed)

    # Use this updated transform in dataset_train
    dataset_train = CocoDataset(
        'train', seed=FLAGS.seed, preload_images=FLAGS.preload_images > 0,  transform=transforms.Compose([Normalizer(), Resizer()])
    )

    dataset_val = CocoDataset('val', seed=0, 
        preload_images=FLAGS.preload_images > 0, 
        transform=transforms.Compose([Normalizer(), Resizer()]))
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, pin_memory=True) 
    
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
    
    # Warmup scheduler
    scheduler1 = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=2000)
    
    # Adjust milestones for MultiStepLR
    milestones = [int(x) for x in FLAGS.lr_step]
    adjusted_milestones = [x - 2000 for x in milestones]
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=adjusted_milestones, gamma=0.1)
    
    # Combined scheduler
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler1, scheduler2], milestones=[2000])
    
    optimizer.zero_grad()
    dataloader_iter = None
    
    times_np, cls_loss_np, bbox_loss_np, total_loss_np = [], [], [], []
    lossFunc = losses.LossFunc()
     
    for i in range(FLAGS.max_iter):
        iter_start_time = time.time()
        
        if dataloader_iter is None or i % len(dataloader_iter) == 0:
            dataloader_iter = iter(dataloader_train)
        
        image, cls, bbox, is_crowd, image_id, _ = next(dataloader_iter)
        
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
        
        # Compute gradients
        total_loss.backward()

        # Gradient clipping
        clip_grad_norm_(model.parameters(), max_norm=5.0)

        # Update parameters
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        # Some logging
        lr = scheduler.get_last_lr()[0]
        total_loss_np.append(total_loss.item())
        cls_loss_np.append(cls_loss.item())
        bbox_loss_np.append(bbox_loss.item())
        times_np.append(time.time() - iter_start_time)
                      
        if (i+1) % log_every == 0:
            print('')
            writer.add_scalar('iteration_rate', len(times_np) / np.sum(times_np), i+1)
            logger('iteration_rate', len(times_np) / np.sum(times_np), i+1)
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
