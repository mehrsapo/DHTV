import torch


class NNDataLoader:
    def __init__(self, data_obj, batch_size):
        self.data = data_obj
        self.batch_size = batch_size

    def get_loader_in_memory(self, inputs, labels):
        data_loader = list(zip(inputs.split(self.batch_size), labels.split(self.batch_size)))

        return data_loader

    @staticmethod
    def shuffle_data_in_memory(inputs, labels):
        permutation_idx = torch.randperm(inputs.size(0))
        inputs = torch.index_select(inputs, 0, permutation_idx)
        labels = torch.index_select(labels, 0, permutation_idx)

        return inputs, labels

    def get_shuffled_train_loader_in_memory(self):
        inputs, labels = self.shuffle_data_in_memory(self.data.train['input'], self.data.train['values'])
        train_loader = self.get_loader_in_memory(inputs, labels)

        return train_loader

    def get_train_valid_loader(self):
        train_loader = self.get_shuffled_train_loader_in_memory()
        valid_loader = self.get_loader_in_memory(self.data.valid['input'], self.data.valid['values'])

        return train_loader, valid_loader

    def get_test_loader(self):
        test_loader = self.get_loader_in_memory(self.data.test['input'], self.data.test['values'])

        return test_loader
