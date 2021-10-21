import copy
import logging
import time
import torch
#import wandb
import os
import numpy as np
from torch import nn

from .utils import transform_list_to_tensor, Saver, EvaluationMetricsKeeper
from torch.utils.tensorboard import SummaryWriter


class FedDetecAggregator(object):
    def __init__(self, worker_num, device, model, args, model_trainer):
        self.trainer = model_trainer
        self.worker_num = worker_num
        self.device = device
        self.args = args
        log_dir=os.path.join(args.log_dir,str(args.client_num_in_total)+'_client_'+str(args.epochs)+'_local_epoch/')
        self.writer=SummaryWriter(log_dir)
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()

        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

        self.train_precision_client_dict = dict()
        self.train_recall_client_dict=dict()
        self.train_mAP_client_dict = dict()
        self.train_loss_client_dict = dict()

        self.test_precision_client_dict = dict()
        self.test_recall_client_dict = dict()
        self.test_mAP_client_dict = dict()
        self.test_loss_client_dict = dict()

        self.best_mAP = 0.
        self.best_mAP_clients = dict()

        self.saver = Saver(args)
        self.saver.save_experiment_config()

        logging.info('Initializing FedDetecAggregator with workers: {0}'.format(worker_num))

    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("Add model index: {}".format(index))
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        start_time = time.time()
        model_list = []
        training_num = 0

        for idx in range(self.worker_num):
            if self.args.is_mobile == 1:
                self.model_dict[idx] = transform_list_to_tensor(self.model_dict[idx])
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]

        logging.info("Aggregating...... {0}, {1}".format(len(self.model_dict), len(model_list)))

        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)

        end_time = time.time()
        logging.info("Aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes: {}".format(client_indexes))
        return client_indexes

    def add_client_test_result(self, round_idx, client_idx, train_eval_metrics: EvaluationMetricsKeeper,
                               test_eval_metrics: EvaluationMetricsKeeper):
        logging.info("Adding client test result : {}".format(client_idx))

        # Populating Training Dictionary
        if round_idx and round_idx % self.args.evaluation_frequency == 0:
            self.train_precision_client_dict[client_idx] = train_eval_metrics.precision
            self.train_recall_client_dict[client_idx] = train_eval_metrics.recall
            self.train_mAP_client_dict[client_idx] = train_eval_metrics.mAP
            self.train_loss_client_dict[client_idx] = train_eval_metrics.loss

        # Populating Testing Dictionary
        self.test_precision_client_dict[client_idx] = test_eval_metrics.precision
        self.test_recall_client_dict[client_idx] = test_eval_metrics.recall
        self.test_mAP_client_dict[client_idx] = test_eval_metrics.mAP
        self.test_loss_client_dict[client_idx] = test_eval_metrics.loss

        if self.args.save_client_model:
            best_mAP = self.best_mAP_clients.setdefault(client_idx, 0.)
            test_mAP = self.test_mAP_client_dict[client_idx]

            if test_mAP > best_mAP:
                self.best_mAP_clients[client_idx] = test_mAP
                logging.info('Saving Model Checkpoint for Client: {0} --> Previous mAP:{1}; Improved mAP:{2}'.format(
                    client_idx, best_mAP, test_mAP))
                is_best = False
                filename = "client" + str(client_idx) + "_checkpoint.pth.tar"
                saver_state = {
                    'best_pred': test_mAP,
                    'round': round_idx + 1,
                    'state_dict': self.model_dict[client_idx]
                }

                test_eval_metrics_dict = {
                    'precision': self.test_precision_client_dict[client_idx],
                    'recall': self.test_recall_client_dict[client_idx],
                    'mAP': self.test_mAP_client_dict[client_idx],
                    'loss': self.test_loss_client_dict[client_idx]
                }

                saver_state['test_data_evaluation_metrics'] = test_eval_metrics_dict

                if round_idx and round_idx % self.args.evaluation_frequency == 0:
                    train_eval_metrics_dict = {
                        'precision': self.train_precision_client_dict[client_idx],
                        'recall': self.train_recall_client_dict[client_idx],
                        'mAP': self.train_mAP_client_dict[client_idx],
                        'loss': self.train_loss_client_dict[client_idx]
                    }
                    saver_state['train_data_evaluation_metrics'] = train_eval_metrics_dict

                self.saver.save_checkpoint(saver_state, is_best, filename)

    def output_global_acc_and_loss(self, round_idx):
        logging.info("################## Output global mAP and loss for round {} :".format(round_idx))

        if round_idx and round_idx % self.args.evaluation_frequency == 0:
            # Test on training set
            train_precision = np.array([self.train_precision_client_dict[k] for k in self.train_precision_client_dict.keys()]).mean()
            train_recall = np.array(
                [self.train_recall_client_dict[k] for k in self.train_recall_client_dict.keys()]).mean()
            train_mAP = np.array([self.train_mAP_client_dict[k] for k in self.train_mAP_client_dict.keys()]).mean()
            train_loss = np.array([self.train_loss_client_dict[k] for k in self.train_loss_client_dict.keys()]).mean()

            # Train Logs
            #wandb.log({"Train/Precision": train_precision, "round": round_idx})
            #wandb.log({"Train/Recall": train_recall, "round": round_idx})
            #wandb.log({"Train/mAP": train_mAP, "round": round_idx})

            #wandb.log({"Train/Loss": train_loss, "round": round_idx})
            stats = {'training_precision': train_precision,
                     'training_recall': train_recall,
                     'training_mAP': train_mAP,
                     'training_loss': train_loss}
            logging.info("Testing statistics: {}".format(stats))

        # Test on testing set
        test_precision = np.array([self.test_precision_client_dict[k] for k in self.test_precision_client_dict.keys()]).mean()
        test_recall = np.array(
            [self.test_recall_client_dict[k] for k in self.test_recall_client_dict.keys()]).mean()
        test_mAP = np.array([self.test_mAP_client_dict[k] for k in self.test_mAP_client_dict.keys()]).mean()
        test_loss = np.array([self.test_loss_client_dict[k] for k in self.test_loss_client_dict.keys()]).mean()

        # Test Logs
        #wandb.log({"Test/Acc": test_precision, "round": round_idx})
        #wandb.log({"Test/Recall": test_recall, "round": round_idx})
        #wandb.log({"Test/mAP": test_mAP, "round": round_idx})

        #wandb.log({"Test/Loss": test_loss, "round": round_idx})
        self.writer.add_scalar('Test/mAP',test_mAP,round_idx)
        stats = {'testing_precision': test_precision,
                 'testing_recall': test_recall,
                 'testing_mAP': test_mAP}

        logging.info("Testing statistics: {}".format(stats))
        if test_mAP > self.best_mAP:
            previous_mAP = self.best_mAP
            self.best_mAP = test_mAP
            #wandb.run.summary["best_mAP"] = self.best_mAP
            #wandb.run.summary["Round Number for best mAP"] = round_idx
            if self.args.save_model:
                logging.info('Saving Model Checkpoint --> Previous mAP:{0}; Improved mAP:{1}'.format(previous_mAP,
                                                                                                       test_mAP))
                is_best = True

                saver_state = {
                    'best_pred': self.best_mAP,
                    'round': round_idx + 1,
                    'state_dict': self.trainer.get_model_params(),
                }

                test_eval_metrics_dict = {
                    'precision': test_precision,
                    'recall': test_recall,
                    'mAP': test_mAP
                }
                saver_state['test_data_evaluation_metrics'] = test_eval_metrics_dict

                if round_idx and round_idx % self.args.evaluation_frequency == 0:
                    train_eval_metrics_dict = {
                        'precision': train_precision,
                        'recall': train_recall,
                        'mIoU': train_mAP,
                        'loss': train_loss
                    }
                    saver_state['train_data_evaluation_metrics'] = train_eval_metrics_dict
                self.saver.save_checkpoint(saver_state, is_best)


