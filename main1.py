"""
This script is the entrance for the entire project.
"""
from getData import main_update
from processBusData import main_process, split_data
from getData import main_update
from geoWeight import main_geoweight
from tf_regression import main_batch_train_and_eval


if __name__ == '__main__':
    # Update the BusStop data through Datamall API
    main_update()

    # Roughly clean and rearrange all the data we have
    main_process()

    # Assign the geographical weight from population and bus volume to data
    main_geoweight()

    # Train the model using different settings in batch, check the training log for evaluation
    main_batch_train_and_eval()

    # Note:--------------------------------------------------------------------
    # Training have to run in python console instead of run directly
    # Some procedures may overwrite each other, please do make backups

