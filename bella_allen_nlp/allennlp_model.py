from typing import Optional, List
import json
import tempfile
from pathlib import Path
import random

from allennlp.common.params import Params
from allennlp.commands.train import train_model_from_file
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.models.archival import load_archive
from bella.data_types import TargetCollection

class AllenNLPModel():
    '''
    This creates a wrapper for all of the Allen NLP models that will be 
    compatible closely with the bella.models.base.BaseModel interface

    To predict I might have to implement a predictor.
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
        self._param_fp = model_param_fp

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
        self.fitted = True

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
            return load_archive(self.save_dir / "model.tar.gz", 
                                cuda_device=cuda_device)
        raise FileNotFoundError('There is nothing at the save dir:\n'
                                f'{self.save_dir.resolve()}')
        
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

        model_param_fp = str(model_param_fp.resolve())
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