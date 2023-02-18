from sklearn.neighbors import KernelDensity
import numpy as np

class KDE:
    # Class to learn esimated probability distribution
    def __init__(
        self,
        data: np.ndarray,
        val_split: float = 0.10,
        test_split: float = 0.05,
        kernel_type: str = "gaussian",
        search_type: str = "logarithmic",
        bw_search: tuple = (0.002, 5),
        N_search: int = 50,
    ):
        """
        data - (N, m) dimensional input. N corresponds to number of samples,
                m corresponds to the dimensionality of the data
        val_split - fraction of the data to use for validation
                to estimate the best bandwidth
        test_split - fraction of the data to use for evaulation
                the model performance at the end
        kernel_type - type of kernel to use for KDE
            available: gaussian, epanechnikov, tophat,
                exponential, linear, cosine
        search_type - either stochastic or deterministic, method for
                searching the bandwidth space
        bw_search - range for searching the best bandwidth for the model
        """
        self.verbose = True

        # If data of shape (N, ) is passed in,
        #       assume user meant 1-d data (N, 1)

        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=1)

        self.data_dimension = data.shape[1]

        # Training the model given the parameters
        self.split_data(data, val_split, test_split)
        self.single_KT_train(kernel_type, bw_search, N_search, search_type)

    def split_data(
        self, data: np.ndarray, val_split: float, test_split: float
    ):

        # code safety check
        if val_split + test_split >= 1:
            raise Exception("No training data with given slicing")
        elif val_split == 0:
            raise Exception(
                "Unable to train bandwidth paramter without validation data"
            )
        np.random.shuffle(data)

        # convert from decimals to array indicies
        val_split_index_start = int((1 - val_split - test_split) * len(data))
        val_split_index_end = int((1 - test_split) * len(data))

        self.testing = False
        if test_split != 0:
            self.testing = True

        # and slice up the data into the three segments
        self.train = data[0:val_split_index_start]
        self.val = data[val_split_index_start:val_split_index_end]
        self.test = data[val_split_index_end:]

    def single_KT_train(
        self,
        kernel_type: str,
        bw_search: tuple,
        N_search: int,
        search_type: str,
    ):
        """
        Execute training over the entire bandwidth space,
            finding the corresponding best kernel
        For a single kernel type

        Inputs:
        Refer to __init__()
        """
        # all of the bandwidths to test
        if search_type == "uniform":
            bw_tests = np.linspace(bw_search[0], bw_search[1], N_search)
        elif search_type == "stochastic":
            bw_tests = np.random.uniform(bw_search[0], bw_search[1], N_search)
        elif search_type == "logarithmic":
            bw_tests = np.logspace(
                np.log2(bw_search[0]), np.log2(bw_search[1]), base=2
            )
        else:
            raise Exception("Invalid search type")
        # place to save validation error for each bandwidth test
        bw_scores = dict()

        # scores for plotting
        ordered_scores = []

        for bandwidth in bw_tests:
            # train the model
            model = self.training_iteration(kernel_type, bandwidth)

            # score the model and append that score
            kernel_score = self.score_kernel(model, "val")
            bw_scores[kernel_score] = model

            ordered_scores.append(kernel_score)

        # choose the best model
        best_model = bw_scores[max(bw_scores)]

        # optionally perform final testing
        if self.testing:
            test_score = self.score_kernel(best_model, "test")
            print(f"Final log likelihood: {test_score:.3f}")
            self.test_score = test_score

        # save the best kernel to be the model kernel
        self.model = best_model

    def training_iteration(self, kernel_type: str, bandwidth: float):
        """
        Obtain a model given paramters
        """
        model = KernelDensity(kernel=kernel_type, bandwidth=bandwidth)
        model.fit(self.train)
        return model

    def score_kernel(self, model, dataset_name: str):
        """
        Score the performance of a model on dataset_name, either
        "val" : validation data
        "test" : testing data
        """
        if dataset_name == "val":
            scores = model.score_samples(self.val)
        elif dataset_name == "test":
            scores = model.score_samples(self.test)
        else:
            raise Exception("Invalid dataset name")

        avg_loss = np.sum(scores) / len(scores)

        return avg_loss

    def get_model(self):
        """
        Fetch the KDE model

        Output:
        KernelDensity model
        """
        return self.model

    def predict(self, data: np.ndarray, convert_ln: bool = True):
        """
        Predict probability of given data

        Inputs:
        data - (N, m) dimensional data on which to perform estimation.
                m must match dimensionality of trained data
        convert_ln - optional argument. Specify whether to provide
                outputs as probability or ln(probability)
        Outputs:
        (N, ) dimensional array corresponding to model
                 prediction of probability
        """
        if data.shape[1] != self.data_dimension:
            raise Exception(
                f"Dims mismatch:, {data.shape[1]}, {self.data_dimension}"
            )

        result = self.model.score_samples(data)

        if convert_ln:  # convert to 0 -> 1 probabilities from log spacels

            return np.exp(result)
        return result

    def sample(self, nsamples: int):
        """
        Sample n samples from the distribution learned by the KDE
        """
        return self.model.sample(nsamples)
        