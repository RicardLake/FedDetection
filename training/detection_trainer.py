import logging, time
import sys, os

import numpy as np
import torch

# add the FedML root directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from FedML.fedml_core.trainer.model_trainer import ModelTrainer
from FedML.fedml_api.distributed.feddetec.utils import  Evaluator, LR_Scheduler, \
    EvaluationMetricsKeeper
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
import utils
import math


class DetectionTrainer(ModelTrainer):
    def __init__(self, model, args=None):
        super(DetectionTrainer, self).__init__(model, args)

    def get_model_params(self):
        if self.args.backbone_freezed:
            logging.info('Initializing model; Backbone Freezed')
            return self.model.encoder_decoder.cpu().state_dict()
        else:
            logging.info('Initializing end-to-end model')
            return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        if self.args.backbone_freezed:
            logging.info('Updating Global model; Backbone Freezed')
            self.model.encoder_decoder.load_state_dict(model_parameters)
        else:
            logging.info('Updating Global model')
            self.model.load_state_dict(model_parameters)

    def train(self, train_data, device):
        model = self.model
        args = self.args
        model.to(device)
        model.train()
        #criterion = SegmentationLosses().build_loss(mode=args.loss_type)
        scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(train_data))

        if args.client_optimizer == "sgd":

            if args.backbone_freezed:
                optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr * 10,
                                            momentum=args.momentum, weight_decay=args.weight_decay,
                                            nesterov=args.nesterov)
            else:
                train_params = [{'params': self.model.parameters(), 'lr': args.lr}]

                optimizer = torch.optim.SGD(train_params, momentum=args.momentum, weight_decay=args.weight_decay,
                                            nesterov=args.nesterov)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                         lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)

        epoch_loss = []

        for epoch in range(args.epochs):
            metric_logger = utils.MetricLogger(delimiter="  ")
            metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            header = 'Epoch: [{}]'.format(epoch)
            t = time.time()
            batch_loss = []
            logging.info('Trainer_ID: {0}, Epoch: {1}'.format(self.id, epoch))
            batch_idx=0
            for images, targets in metric_logger.log_every(train_data, 20, header):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                scheduler(optimizer, batch_idx, epoch)

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                loss_value = losses_reduced.item()

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    print(loss_dict_reduced)
                    sys.exit(1)
                    
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                batch_loss.append(losses.item())

                metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
                batch_idx+=1

    #def train(self, train_data, device):
    #    model = self.model
    #    args = self.args
    #    model.to(device)
    #    model.train()
        #criterion = SegmentationLosses().build_loss(mode=args.loss_type)
    #    scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(train_data))

      #  if args.client_optimizer == "sgd":

        #    if args.backbone_freezed:
       #         optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr * 10,
         #                                   momentum=args.momentum, weight_decay=args.weight_decay,
         #                                   nesterov=args.nesterov)
         #   else:
         #       train_params = [{'params': self.model.parameters(), 'lr': args.lr}]

          #      optimizer = torch.optim.SGD(train_params, momentum=args.momentum, weight_decay=args.weight_decay,
          #                                  nesterov=args.nesterov)
        #else:
        #    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
           #                              lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)

        #epoch_loss = []

        #for epoch in range(args.epochs):
        #    t = time.time()
        #    batch_loss = []
        #    logging.info('Trainer_ID: {0}, Epoch: {1}'.format(self.id, epoch))

        #    for (batch_idx, batch) in enumerate(train_data):
        #        images = list(image.to(device) for image in images)
        #        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #        loss_dict = model(images, targets)
        #        losses = sum(loss for loss in loss_dict.values())
        #        scheduler(optimizer, batch_idx, epoch)
        #        optimizer.zero_grad()
        #        losses.backward()
        #        optimizer.step()
        #        batch_loss.append(losses.item())
        #        if (batch_idx % 100 == 0):
        #            logging.info(
        #                'Trainer_ID: {0} Iteration: {1}, Loss: {2}, Time Elapsed: {3}'.format(self.id, batch_idx, losses,
        #                                                                                      (time.time() - t) / 60))

        #    if len(batch_loss) > 0:
        #        epoch_loss.append(sum(batch_loss) / len(batch_loss))
        #        logging.info('(Trainer_ID: {}. Local Training Epoch: {} \tLoss: {:.6f}'.format(self.id,
        #                                                                                       epoch,
        #                                                                                       sum(epoch_loss) / len(
        #                                                                                           epoch_loss)))

    def test(self, test_data, device):
        logging.info("Evaluating on Trainer ID: {}".format(self.id))
        model = self.model
        args = self.args
        n_threads = torch.get_num_threads()
        model.eval()
        model.to(device)
        cpu_device = torch.device("cpu")
        t = time.time()
        coco = get_coco_api_from_dataset(test_data.dataset)
        iou_types = ['bbox']
        coco_evaluator = CocoEvaluator(coco, iou_types)
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test:'
        test_precision = test_recall = test_mAP = test_loss = test_total = 0.

        with torch.no_grad():
            for images, targets in metric_logger.log_every(test_data, 100, header):
                images = list(img.to(device) for img in images)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                model_time = time.time()
                outputs = model(images)

                outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
                model_time = time.time() - model_time
                res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
                evaluator_time = time.time()
                coco_evaluator.update(res)
                evaluator_time = time.time() - evaluator_time
                metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

                # time_end_test_per_batch = time.time()
                # logging.info("time per batch = " + str(time_end_test_per_batch - time_start_test_per_batch))
                # logging.info("Client = {0} Batch = {1}".format(self.client_index, batch_idx)
        # Evaluation Metrics (Averaged over number of samples)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        coco_evaluator.synchronize_between_processes()

        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        torch.set_num_threads(n_threads)
        test_mAP=coco_evaluator.coco_eval['bbox'].stats[0]
        test_mAP50=coco_evaluator.coco_eval['bbox'].stats[1]
        test_mAP75=coco_evaluator.coco_eval['bbox'].stats[2]
        test_mAP_small=coco_evaluator.coco_eval['bbox'].stats[3]
        test_mAP_medium=coco_evaluator.coco_eval['bbox'].stats[4]
        test_mAP_large=coco_evaluator.coco_eval['bbox'].stats[4]
        eval_metrics = EvaluationMetricsKeeper(test_precision, test_recall, test_mAP,test_loss)
        return eval_metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None):
        pass

