from collections import deque
from typing import Iterable, Deque
import logging
import random

from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset import Batch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DataIterator.register("augmented")
class AugmentedIterator(DataIterator):
    """
    Augmented iterator that takes a dataset and selects the data to return 
    based on the epoch number associated to the samples in the dataset and the 
    epoch the iterator is currently on. It can also shuffle the dataset. This 
    requires the dataset to be pre-processed and have a field called 
    `epoch_to_be_selected_on` where that field contains a list of unique numbers 
    where each number would suggest which epoch that sample will be within. 
    This iterator will then generate fixed sized batches. It takes the same 
    parameters as :class:`allennlp.data.iterators.DataIterator`

    It also overrides the `get_num_batches` function as the whole dataset is 
    never used only a sub-sample is used based on the the samples chosen at 
    each epoch.
    """
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        # First break the dataset into memory-sized lists:
        epoch_number = self._epochs[id(instances)]
        filtered_instances = []
        for instance in instances:
            if 'epoch_numbers' in instance.fields:
                epoch_numbers = instance.fields['epoch_numbers'].array
                if epoch_number in epoch_numbers:
                    filtered_instances.append(instance)
                    continue
                elif -1 in epoch_numbers:
                    filtered_instances.append(instance)
                    continue
                else:
                    continue
            filtered_instances.append(instance)
        for instance_list in self._memory_sized_lists(filtered_instances):
            if shuffle:
                random.shuffle(instance_list)
            iterator = iter(instance_list)
            excess: Deque[Instance] = deque()
            # Then break each memory-sized list into batches.
            for batch_instances in lazy_groups_of(iterator, self._batch_size):
                for possibly_smaller_batches in self._ensure_batch_is_sufficiently_small(batch_instances, excess):
                    batch = Batch(possibly_smaller_batches)
                    yield batch
            if excess:
                yield Batch(excess)


    def get_num_batches(self, instances: Iterable[Instance]) -> int:
        """
        The way this needs to be changed is that we could iterate through the 
        instances and count the number in the current epoch? Don't know 
        how slow this would be through?

        Returns the number of batches that ``dataset`` will be split into; if you want to track
        progress through the batch with the generator produced by ``__call__``, this could be
        useful.
        """
        return 1
    #    epoch_number = self._epochs[id(instances)]
    #    if is_lazy(instances) and self._instances_per_epoch is None:
    #        # Unable to compute num batches, so just return 1.
    #        return 1
    #    elif self._instances_per_epoch is not None:
    #        return math.ceil(self._instances_per_epoch / self._batch_size)
    #    else:
    #        # Not lazy, so can compute the list length.
    #        return math.ceil(len(ensure_list(instances)) / self._batch_size)