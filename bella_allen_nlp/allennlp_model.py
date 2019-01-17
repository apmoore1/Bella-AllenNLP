import json
import tempfile
from pathlib import Path

from bella.data_types import TargetCollection

class AllenNLPModel():
    '''
    This creates a wrapper for all of the Allen NLP models that will be 
    compatible closely comptabile with the bella.models.base.BaseModel 
    interface
    '''

    def __init__(self, name: str) -> None:
        '''
        :param name: Name of the model e.g. TDLSTM or IAN
        '''

        self.name = name
        self._fitted = False

    def fit(self, train_data: TargetCollection, val_data: TargetCollection
            ) -> None:
        train_data.to_json_file()

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