from typing import Optional, List, Any, Generator, Dict
import json
import tempfile
from pathlib import Path
import random

from allennlp.common.params import Params
from allennlp.commands.train import train_model_from_file
from allennlp.commands.find_learning_rate import find_learning_rate_model
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.models.archival import load_archive
from bella.data_types import TargetCollection
import numpy as np

from bella_allen_nlp.predictors.target_predictor import TargetPredictor

class AllenNLPModel():
    '''
    This creates a wrapper for all of the Allen NLP models that will be 
    compatible closely with the bella.models.base.BaseModel interface

    I think what needs to be created is a predictor.
    '''

    def __init__(self, name: str, model_param_fp: Path, 
                 save_dir: Optional[Path] = None) -> None:
        '''
        :param name: Name of the model e.g. TDLSTM or IAN
        :param model_params_fp: File path to the model parameters that will 
                                define the AllenNLP model and how to train it.
        :param save_dir: Directory to save the model to. This has to be set
                         up front as the fit function saves the model each 
                         epoch.
        '''

        self.name = name
        self.model = None
        self.save_dir = save_dir
        self._fitted = False
        self._param_fp = model_param_fp.resolve()
        self.labels = None

    def fit(self, train_data: TargetCollection, val_data: TargetCollection,
            test_data: Optional[TargetCollection] = None) -> None:
        '''
        Given the training, validation, and optionally the test data it will 
        train the model that is defined in the model params file provided as 
        argument to the constructor of the class. Once trained the model can 
        be accessed through the `model` attribute.

        NOTE: If the test data is given the model only uses it to fit to the 
        vocabularly that is within the test data, the model NEVER trains on 
        the test data.

        :param train_data: Training data.
        :param val_data: Validation data.
        :param test_data: Optional, test data.
        '''

        model_params = self._preprocess_and_load_param_file(self._param_fp)
        # Ensures that a different random seed is used each time
        self._set_random_seeds(model_params)
        with tempfile.TemporaryDirectory() as temp_dir:
            train_fp = Path(temp_dir, 'train_data.json')
            val_fp = Path(temp_dir, 'val_data.json')

            self._data_to_json(train_data, train_fp)
            self._data_to_json(val_data, val_fp)
            if test_data:
                test_fp = Path(temp_dir, 'test_data.json')
                self._data_to_json(test_data, test_fp)
                self._add_dataset_paths(model_params, train_fp, val_fp, test_fp)
                model_params["evaluate_on_test"] = True
            else:
                self._add_dataset_paths(model_params, train_fp, val_fp)

            save_dir = self.save_dir if self.save_dir else Path(temp_dir, 
                                                                'temp_save_dir')
            
            temp_param_fp = Path(temp_dir, 'temp_param_file.json')
            model_params.to_file(temp_param_fp.resolve())
            trained_model = train_model_from_file(temp_param_fp, save_dir)
            self.model = trained_model
            self.labels = self._get_labels()
        self.fitted = True

    def _predict_iter(self, data: TargetCollection
                      ) -> Generator[Dict[str, Any], None, None]:
        '''
        Iterates over the predictions and yields one prediction at a time.

        This is a useful wrapper as it performs the data pre-processing and 
        assertion checks.

        :param data: Data to predict on
        :yields: A dictionary containing `class_probabilities` and `label`.
        '''
        no_model_error = 'There is no model to make predictions, either fit '\
                         'or load a model.'
        assert self.model, no_model_error
        self.model.eval()

        all_model_params = Params.from_file(self._param_fp)

        reader_params = all_model_params.get("dataset_reader")
        dataset_reader = DatasetReader.from_params(reader_params)
        predictor = TargetPredictor(self.model, dataset_reader)

        batch_size = 64
        if 'iterator' in all_model_params:
            iter_params = all_model_params.get("iterator")
            if 'batch_size' in iter_params:
                batch_size = iter_params['batch_size']
        
        json_data = data.data_dict()
        # Reference
        # https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
        for i in range(0, len(json_data), batch_size):
            json_data_batch = json_data[i:i + batch_size]
            predictions = predictor.predict_batch_json(json_data_batch)
            for prediction in predictions:
                yield prediction

    def predict(self, data: TargetCollection) -> np.ndarray:
        '''
        Given the data to predict with return a matrix of shape 
        [n_samples, n_classes] where the predict class will be one and all 
        others 0.

        To get the class label for these predictions use the `labels` attribute.
        The index of the predicted class is associated to the index within the 
        `labels` attribute.

        :param data: Data to predict on.
        :returns: A matrix of shape [n_samples, n_classes]
        '''
        predictions = self._predict_iter(data)

        n_samples = len(data)
        n_classes = len(self.labels)
        predictions_matrix = np.zeros((n_samples, n_classes))
        for index, prediction in enumerate(predictions):
            class_probabilities = prediction['class_probabilities']
            class_label = np.argmax(class_probabilities)
            predictions_matrix[index][class_label] = 1
        return predictions_matrix

    def predict_label(self, data: TargetCollection, 
                      mapper: Optional[Dict[str, Any]] = None) -> np.ndarray:
        '''
        Given the data to predict with return a vector of class labels.

        Optionally a mapper dictionary can be given to map the class labels 
        to a different label e.g. {'positive': 1, 'neutral': 0, 'negative': -1}.
        
        :param data: Data to predict on.
        :returns: A vector of shape [n_samples]
        '''
        predictions = self._predict_iter(data)

        predictions_list = []
        for prediction in predictions:
            label = prediction['label']
            if mapper:
                label = mapper[label]
            predictions_list.append(label)
        return np.array(predictions_list)

    def probabilities(self, data: TargetCollection) -> np.ndarray:
        '''
        Returns the probability for each class for every sample in the data. 
        The returned matrix is of shape [n_samples, n_classes]

        :param data: Data to predict on
        :returns: probabilities that a class is true for each class for each 
                  sample. 
        '''
        
        predictions = self._predict_iter(data)

        n_samples = len(data)
        n_classes = len(self.labels)
        probability_matrix = np.zeros((n_samples, n_classes))
        for index, prediction in enumerate(predictions):
            class_probabilities = prediction['class_probabilities']
            probability_matrix[index] = class_probabilities
        return probability_matrix

    def load(self, cuda_device: int = -1) -> Model:
        '''
        Loads the model. This does not require you to train the model if the 
        `save_dir` attribute is pointing to a folder containing a trained model.

        This is just a wrapper around the `load_archive` function.

        :param cuda_device: Whether the loaded model should be loaded on to the 
                            CPU (-1) or the GPU (0). Default CPU.
        :returns: The model that was saved at `self.save_dir` 
        '''

        save_dir_err = 'Save directory was not set in the constructor of the class'
        assert self.save_dir, save_dir_err
        if self.save_dir.exists():
            archive = load_archive(self.save_dir / "model.tar.gz", 
                                   cuda_device=cuda_device)
            self.model = archive.model
            self.labels = self._get_labels()
            return self.model
        raise FileNotFoundError('There is nothing at the save dir:\n'
                                f'{self.save_dir.resolve()}')

    def find_learning_rate(self, train_data: TargetCollection,
                           results_dir: Path, 
                           find_lr_kwargs: Optional[Dict[str, Any]] = None
                           ) -> None:
        '''
        Given the training data it will plot learning rate against loss to allow 
        you to find the best learning rate.

        This is just a wrapper around 
        allennlp.commands.find_learning_rate.find_learning_rate_model method.

        :param train_data: Training data.
        :param results_dir: Directory to store the results of the learning rate
                            findings.
        :param find_lr_kwargs: Dictionary of keyword arguments to give to the 
                               allennlp.commands.find_learning_rate.find_learning_rate_model
                               method.
        '''
        model_params = self._preprocess_and_load_param_file(self._param_fp)
        with tempfile.TemporaryDirectory() as temp_dir:
            train_fp = Path(temp_dir, 'train_data.json')
            self._data_to_json(train_data, train_fp)
            model_params['train_data_path'] = str(train_fp.resolve())
            if find_lr_kwargs is None:
                find_learning_rate_model(model_params, results_dir)
            else:
                find_learning_rate_model(model_params, results_dir, **find_lr_kwargs)

    def _get_labels(self) -> List[Any]:
        '''
        Will return all the possible class labels that the model attribute 
        can generate.

        :returns: List of possible labels the model can generate
        '''
        vocab = self.model.vocab
        label_dict = vocab.get_index_to_token_vocabulary('labels')
        return [label_dict[i] for i in range(len(label_dict))]
        
    @staticmethod
    def _data_to_json(data: TargetCollection, file_path: Path) -> None:
        '''
        Converts the data into json format and saves it to the given file path. 
        The AllenNLP models read the data from json formatted files.

        :param data: data to be saved into json format.
        :param file_path: file location to save the data to.
        '''
        target_data = data.data_dict()
        with file_path.open('w+') as json_file:
            for index, data in enumerate(target_data):
                if 'epoch_number' in data:
                    data['epoch_number'] = list(data['epoch_number'])
                json_encoded_data = json.dumps(data)
                if index != 0:
                    json_encoded_data = f'\n{json_encoded_data}'
                json_file.write(json_encoded_data)

    @staticmethod
    def _preprocess_and_load_param_file(model_param_fp: Path) -> Params:
        '''
        Given a model parameter file it will load it as a Params object and 
        remove all data fields for the Param object so that these keys can be 
        added with different values associated to them.

        fields (keys) that are removed:
        1. train_data_path
        2. validation_data_path
        3. test_data_path
        4. evaluate_on_test

        :param model_param_fp: File path to the model parameters that will 
                               define the AllenNLP model and how to train it.
        :returns: The model parameter file as a Params object with the data 
                   fields removed if they exisited.
        '''

        model_param_fp = str(model_param_fp)
        fields_to_remove = ['train_data_path', 'validation_data_path', 
                            'test_data_path', 'evaluate_on_test']
        model_params = Params.from_file(model_param_fp)
        for field in fields_to_remove:
            if field in model_params:
                model_params.pop(field)
        return model_params

    @staticmethod
    def _add_dataset_paths(model_params: Params, train_fp: Path, val_fp: Path, 
                           test_fp: Optional[Path] = None) -> None:
        '''
        Give model parameters it will add the given train, validation and 
        optional test dataset paths to the model parameters.

        Does not return anything as the model parameters object is mutable

        :param model_params: model parameters to add the dataset paths to
        :param train_fp: Path to the training dataset
        :param val_fp: Path to the validation dataset
        :param test_fp: Optional path to the test dataset
        '''

        model_params['train_data_path'] = str(train_fp.resolve())
        model_params['validation_data_path'] = str(val_fp.resolve())
        if test_fp:
            model_params['test_data_path'] = str(test_fp.resolve())

    @staticmethod
    def _set_random_seeds(model_params: Params) -> None:
        '''
        This ensures to some extent that the experiments are NOT reproducible 
        so that we can take into account the random seed problem.

        Returns nothing as the model_params will be modified as they are a 
        mutable object.

        :param model_params: The parameters of the model
        '''

        seed, numpy_seed, torch_seed = [random.randint(1,99999) 
                                        for i in range(3)]
        model_params["random_seed"] = seed
        model_params["numpy_seed"] = numpy_seed
        model_params["pytorch_seed"] = torch_seed     

    def __repr__(self) -> str:
        '''
        :returns: the name of the model e.g. TDLSTM or IAN
        '''
        return self.name

    @property
    def fitted(self) -> bool:
        '''
        If the model has been fitted (default False)

        :return: True or False
        '''

        return self._fitted

    @fitted.setter
    def fitted(self, value: bool) -> None:
        '''
        Sets the fitted attribute

        :param value: The value to assign to the fitted attribute
        '''

        self._fitted = value

    # Not required anymore
    @staticmethod
    def _get_vocab(dataset_reader: DatasetReader, data_paths: List[Path]
                   ) -> Vocabulary:
        '''
        It will create a vocabulary object from the dataset reader given and 
        the data files provided in the data_paths list.

        :param dataset_reader: A dataset reader than can read the data provided 
                               in the data paths list.
        :param data_paths: A list of paths to the data to be read e.g. a list 
                           that contains the training data file path, validation 
                           data file path, and the test data file path.
        :returns: A vocabulary object from the data and dataset reader.
        '''
        all_instances = []
        for data_path in data_paths:
            all_instances.extend(list(dataset_reader.read(data_path)))
        return Vocabulary.from_instances(all_instances)