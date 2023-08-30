"""
These functions are used to repair an unsafe DNN with respect to a set of safety properties.

Authors: Xiaodong Yang, xiaodong.yang@vanderbilt.edu
License: BSD 3-Clause

"""

import time
import logging
from veritex.networks.ffnn import FFNN
import torch
import numpy as np
import copy as cp
import multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
from veritex.methods.worker import Worker
from veritex.methods.shared import SharedState
import torch.optim as optim


class REPAIR:
    """
    A class for the repair of a neural network

        Attributes:
            properties (Property): Safety properties of the neural network
            corrections (function): Functions to approximate the closest safe point to an unsafe point
            output_limit (int): Maximal unsafe data computed in the reachability analysis
            torch_model (Pytorch): Network model in Pytorch
            data (DATA): data to train, validate and test the network

        Methods:
            compute_unsafe_data():
                Compute a set of unsafe data pairs (x,y)s which are entire or subsets of vertices of the unsafe input-output domains.
            generate_data():
                Generate random data for training, validating and testing the network model.
            purify_data():
                Remove the unsafe data from the data
            correct_unsafe_data(unsafe_data):
                Approximate the closest safe data for the unsafe date.
            compute_deviation(model):
                Compute the parameter deviation
            compute_accuracy(model):
                Compute the accuracy of the network model on the test data.
            repair_model_regular(optimizer, loss_fun, alpha, beta, savepath, iters=100, batch_size=200, epochs=200):
                Repair the network model for regression
            repair_model_classification(self, optimizer, loss_fun, alpha, beta, savepath, iters=100, batch_size=2000, epochs=200):
                Repair the network model for classification
    """

    def __init__(self, torch_model, properties_repair, data=None, output_limit=1000):
        """
        Constructs all the necessary attributes for the Repair object

        Parameters:
            torch_model (Pytorch): A network model
            properties_repair (list): Safety properties and functions to correct unsafe elements.
            data (list): Data for training, validating and testing the network model
            output_limit (int): Maximal unsafe data computed in the reachability analysis
        """

        self.properties = [item[0] for item in properties_repair]
        self.corrections = [item[1] for item in properties_repair]
        self.output_limit = output_limit
        self.torch_model = torch_model
        if data is not None:
            self.data = data
        else:
            self.data = self.generate_data()


    def compute_unsafe_data(self):
        """
        Compute a set of unsafe data pairs (x,y)s which are entire or subsets of vertices of the unsafe input-output domains.

        Returns:
             all_unsafe_data (list): Unsafe data pairs (x,y)s over each safety property
        """

        self.ffnn = FFNN(self.torch_model, repair=True)
        all_unsafe_data = []
        num_processors = mp.cpu_count()
        for n, prop in enumerate(self.properties):
            self.ffnn.set_property(prop)
            processes = []
            unsafe_data = []
            shared_state = SharedState(prop, num_processors)
            one_worker = Worker(self.ffnn, output_len=self.output_limit)
            for index in range(num_processors):
                p = mp.Process(target=one_worker.main_func, args=(index, shared_state))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            while not shared_state.outputs.empty():
                unsafe_data.append(shared_state.outputs.get())

            all_unsafe_data.append(unsafe_data)

        return all_unsafe_data


    def generate_data(self, num=10000):
        """
        Generate random data for training, validating and testing the network model.

        Parameters:
            num (int): Total number of the data

        Returns:
            data (DATA): Data for training, validating and testing the network model
        """
        lbs = self.properties[0].input_ranges[0]
        ubs = self.properties[0].input_ranges[1]

        train_x = torch.tensor([np.random.uniform(lbs[i], ubs[i], num).tolist() for i in range(len(lbs))]).T
        with torch.no_grad():
            train_y = self.torch_model(train_x)
        train_data = self.purify_data([train_x, train_y])

        valid_x = torch.tensor([np.random.uniform(lbs[i], ubs[i], int(num * 0.5)).tolist() for i in range(len(lbs))]).T
        with torch.no_grad():
            valid_y = self.torch_model(valid_x)
        valid_data = self.purify_data([valid_x, valid_y])

        test_x = torch.tensor([np.random.uniform(lbs[i], ubs[i], int(num * 0.5)).tolist() for i in range(len(lbs))]).T
        with torch.no_grad():
            test_y = self.torch_model(test_x)
        test_data = self.purify_data([test_x, test_y])

        return DATA(train_data, valid_data, test_data)


    def purify_data(self, data):
        """
        Remove the unsafe data from the data

        Parameters:
            data (np.ndarray): Data pairs (x,y)s

        Returns:
            data_x (np.ndarray): Safe data x on all safety properties
            data_y (np.ndarray): safe data y on all safety properties
        """
        data_x = data[0]
        data_y = data[1]
        for p in self.properties:
            lb, ub = p.lbs, p.ubs
            for ufd in p.unsafe_domains:
                M, vec = torch.tensor(ufd[0], dtype=torch.float32), torch.tensor(ufd[1],dtype=torch.float32)
                bools = torch.ones(len(data_x), dtype=torch.bool)
                for n in range(len(lb)):
                    lbx, ubx = lb[n], ub[n]
                    x = data_x[:, n]
                    bools = (x > lbx) & (x < ubx) & bools

                if not torch.any(bools):
                    continue
                outs = torch.mm(M, data_y.T) + vec
                out_bools = torch.all(outs<=0, dim=0) & bools
                if not torch.any(out_bools):
                    continue

                safe_indx = torch.nonzero(~out_bools)[:,0]
                data_x = data_x[safe_indx]
                data_y = data_y[safe_indx]

        return [data_x, data_y]


    def correct_unsafe_data(self, unsafe_data):
        """
        Approximate the closest safe data for the unsafe date.

        Parameters:
            unsafe_data (list): Unsafe data pairs (x,y) over each safety property

        Returns:
            original_Xs (Tensor): Original data x
            corrected_Ys (Tensor): Corrected or closest safe y for the unsafe y
        """

        length_unsafe_data = 0
        corrected_Ys = []
        original_Xs = []
        for n, subdata in enumerate(unsafe_data):
            length_unsafe_data += len(subdata)
            if len(subdata) == 0:
                continue

            correction = self.corrections[n]
            safe_x, safe_y = correction(subdata)
            original_Xs.append(safe_x)
            corrected_Ys.append(safe_y)

        corrected_Ys = torch.cat(corrected_Ys, dim=0)
        original_Xs = torch.cat(original_Xs, dim=0)
        return original_Xs, corrected_Ys


    def compute_deviation(self, model):
        """
        Compute the parameter deviation

        Parameters:
            model (Pytorch): A network model

        Returns:
            rls (float): Parameter deviation
        """

        model.eval()
        with torch.no_grad():
            predicts = model(self.data.test_data[0]) # The minimum is the predication
        actl_ys = self.data.test_data[1]
        rls = torch.sqrt(torch.sum(torch.square(predicts-actl_ys)))/ torch.sqrt(torch.sum(torch.square(actl_ys)))
        logging.info(f'Output deviation on the test data: {rls :.2f} ')
        return rls


    def compute_accuracy(self, model):
        """
        Compute the accuracy of the network model on the test data.

        Parameters:
            model (Pytorch): A network model

        Returns:
            accuracy (float): Accuracy
        """

        model.eval()
        with torch.no_grad():
            predicts = model(self.data.test_data[0]) * (-1)  # The minimum is the predication
        pred_actions = torch.argmax(predicts, dim=1)
        actl_actions = torch.argmax(self.data.test_data[1] * (-1), dim=1)
        actions_times = torch.tensor([len(torch.nonzero(actl_actions==n)[:,0]) for n in range(predicts.shape[1])])
        self.optimal_dim = torch.argmax(actions_times)
        accuracy = len(torch.nonzero(pred_actions == actl_actions)) / len(predicts)
        logging.info(f'Accuracy on the test data: {accuracy * 100 :.2f}% ')
        return accuracy


    def repair_model_regular(self, optimizer, loss_fun, alpha, beta, savepath, iters=100, batch_size=200, epochs=200):
        """
        Repair the network model for regression

        Parameters:
            optimizer (optimizer): Optimizer for the training of the model
            loss_fun (function): Loss function for the training of the model
            alpha (float): Weight of the distance between safe domains and unsafe domains in the loss function
            beta (float): Wight of the loss value on training data in the loss function
            savepath (str): Path to save the repaired network
            iters (int): Number of the iterations
            batch_size (int): Batch size
            epochs (int): Number of epochs

        """

        t0 = time.time()
        all_test_deviation = []
        repaired = False
        for num in range(iters):
            logging.info(f'Iteration of repair: {num}')
            # deviation = self.compute_deviation(self.torch_model)
            # all_test_deviation.append(deviation)

            # Compute unsafe domain of the network and construct corrected safe training data
            tt0 = time.time()
            unsafe_data = self.compute_unsafe_data()
            logging.info(f'Time for reachability analysis: {time.time()-tt0 :.2f} sec')
            if np.all([len(sub)==0 for sub in unsafe_data]):
                logging.info('The accurate and safe candidate model is found? True')
                logging.info(f'Total running time: {time.time()-t0 :.2f} sec')
                torch.save(self.torch_model, savepath + "/repaired_model.pt")
                repaired = True
                break

            if not np.all([len(sub)==0 for sub in unsafe_data]):
                original_Xs, corrected_Ys = self.correct_unsafe_data(unsafe_data)
                train_x = original_Xs
                train_y = corrected_Ys
            else:
                train_x = self.data.train_data[0]
                train_y = self.data.train_data[1]

            training_dataset_adv = TensorDataset(train_x, train_y)
            train_loader_adv = DataLoader(training_dataset_adv, batch_size, shuffle=True)
            training_dataset_train = TensorDataset(self.data.train_data[0], self.data.train_data[1])
            train_loader_train = DataLoader(training_dataset_train, batch_size, shuffle=True)

            logging.info('Start retraining for the repair...')
            self.torch_model.train()
            for e in range(epochs):
                # print('\rEpoch: '+str(e)+'/'+str(epochs),end='')
                for batch_idx, data in enumerate(zip(train_loader_adv, train_loader_train)):
                    datax, datay = data[0][0], data[0][1]
                    datax_train, datay_train = data[1][0], data[1][1]
                    optimizer.zero_grad()

                    predicts_adv = self.torch_model(datax)
                    loss_adv = loss_fun(datay, predicts_adv)
                    predicts_train = self.torch_model(datax_train)
                    loss_train = loss_fun(datay_train, predicts_train)
                    loss = alpha*loss_adv + beta*loss_train
                    loss.backward()
                    optimizer.step()
            self.torch_model.cpu()
            logging.info('The retraining is done\n')
            if num % 1 == 0:
                torch.save(self.torch_model, savepath + "/epoch" + str(num) + ".pt")

        if not repaired:
            logging.info('The accurate and safe candidate model is found? False')
            logging.info(f'Total running time: {time.time() - t0 :.2f} sec')
            torch.save(self.torch_model, savepath + "/unrepaired_model.pt")


    def repair_model_classification(self, optimizer, loss_fun, alpha, beta, savepath, iters=100, batch_size=2000, epochs=200):
        """
        Repair the network model for classification

        Parameters:
            optimizer (optimizer): Optimizer for the training of the model
            loss_fun (function): Loss function for the training of the model
            alpha (float): Weight of the distance between safe domains and unsafe domains in the loss function
            beta (float): Wight of the loss value on training data in the loss function
            savepath (str): Path to save the repaired network
            iters (int): Number of the iterations
            batch_size (int): Batch size
            epochs (int): Number of epochs

        """
        all_test_accuracy = []
        accuracy_old = 1.0
        candidate_old = cp.deepcopy(self.torch_model)
        reset_flag = False
        repaired = False
        t0 = time.time()
        for num in range(iters):
            logging.info(f'Iteration of repair: {num}')
            accuracy_new = self.compute_accuracy(self.torch_model)
            # Restore the previous model if there is a large drop of accuracy in the current model
            if accuracy_old - accuracy_new > 0.1:
                logging.info('A large drop of accuracy!')
                self.torch_model = cp.deepcopy(candidate_old)
                # Decrease the learning rate to reduce the accuracy degradation
                lr = optimizer.param_groups[0]['lr'] * 0.8
                logging.info(f'Current lr: {lr}')
                optimizer = optim.SGD(self.torch_model.parameters(), lr=lr, momentum=0.9)
                reset_flag = True
                continue

            # Compute unsafe domain of the network and construct corrected safe training data
            if not reset_flag:
                candidate_old = cp.deepcopy(self.torch_model)
                accuracy_old = accuracy_new
                all_test_accuracy.append(accuracy_new)
                tt0 = time.time()
                unsafe_data = self.compute_unsafe_data()
                logging.info(f'Time for reachability analysis: {time.time()-tt0 :.2f} sec')
                if np.all([len(sub)==0 for sub in unsafe_data]):
                    logging.info('The accurate and safe candidate model is found? True')
                    logging.info(f'Total running time: {time.time()-t0 :.2f} sec')
                    torch.save(self.torch_model, savepath + "/repaired_model.pt")
                    repaired = True
                    break

                if not np.all([len(sub)==0 for sub in unsafe_data]):
                    original_Xs, corrected_Ys = self.correct_unsafe_data(unsafe_data)
                    train_x = original_Xs
                    train_y = corrected_Ys
                else:
                    train_x = self.data.train_data[0]
                    train_y = self.data.train_data[1]

                training_dataset_adv = TensorDataset(train_x, train_y)
                train_loader_adv = DataLoader(training_dataset_adv, batch_size, shuffle=True)
                training_dataset_train = TensorDataset(self.data.train_data[0], self.data.train_data[1])
                train_loader_train = DataLoader(training_dataset_train, batch_size, shuffle=True)

            reset_flag = False
            logging.info('Start retraining for the repair...')
            self.torch_model.train()
            for e in range(epochs):
                # print('\rEpoch: '+str(e)+'/'+str(epochs),end='')
                for batch_idx, data in enumerate(zip(train_loader_adv, train_loader_train)):
                    datax, datay = data[0][0], data[0][1]
                    datax_train, datay_train = data[1][0], data[1][1]
                    optimizer.zero_grad()

                    predicts_adv = self.torch_model(datax)
                    loss_adv = loss_fun(datay, predicts_adv)
                    predicts_train = self.torch_model(datax_train)
                    loss_train = loss_fun(datay_train, predicts_train)
                    loss = alpha*loss_adv + beta*loss_train
                    loss.backward()
                    optimizer.step()

            self.torch_model.cpu()
            logging.info('The retraining is done\n')
            if num % 1 == 0:
                torch.save(self.torch_model, savepath + "/acasxu_epoch" + str(num) + ".pt")

        if not repaired:
            logging.info('The accurate and safe candidate model is found? False')
            logging.info(f'Total running time: {time.time() - t0 :.2f} sec')
            torch.save(self.torch_model, savepath + "/unrepaired_model.pt")


class DATA:
    """
    A class for data

    Attributes:
        train_data (np.ndarray): Training data
        valid_data (np.ndarray): Validation data
        test_data (np.ndarray): Test data
    """
    def __init__(self, train_data, valid_data, test_data):
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

