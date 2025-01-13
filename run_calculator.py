#TODO: Add the run_calculator.py file
from calculators.Storage import Storage
from calculators.Transmission_simple import Transmission_simple
from calculators.DataPreprocessing import DataPreprocessing
from calculators.Inference import Inference
from calculators.Training import Training
import calculator_config as cfg

def calculate_total_energy():
    # Initialize calculators
    transmission = Transmission_simple(
        failure_rate=cfg.FAILURE_RATE,
        application=cfg.APPLICATION_PROTOCOLS,
        presentation=cfg.PRESENTATION_PROTOCOLS,
        session=cfg.SESSION_PROTOCOLS,
        transport=cfg.TRANSPORT_PROTOCOLS,
        network=cfg.NETWORK_PROTOCOLS,
        datalink=cfg.DATALINK_PROTOCOLS,
        physical=cfg.PHYSICAL_PROTOCOLS
    )
    storage = Storage(
        storage_type=cfg.STORAGE_TYPE,
        raid_level=cfg.RAID_LEVEL,
        num_disks=cfg.NUM_DISKS
    )
    preprocessing = DataPreprocessing(
        preprocessing_type=cfg.PREPROCESSING_TYPE,
        processor_flops_per_second=cfg.PROCESSOR_FLOPS_PER_SECOND,
        processor_max_power=cfg.PROCESSOR_MAX_POWER,
    )
    training = Training(
        model_name=cfg.MODEL_NAME,
        num_epochs=cfg.NUM_EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        processor_flops_per_second=cfg.PROCESSOR_FLOPS_PER_SECOND,
        processor_max_power=cfg.PROCESSOR_MAX_POWER,
        num_samples=cfg.DATA_SIZE,
        input_size=cfg.INPUT_SIZE,
        evaluation_strategy=cfg.EVALUATION_STRATEGY,
        k_folds=cfg.K_FOLDS,
        split_ratio=cfg.SPLIT_RATIO
    )
    
    inference = Inference(
        model_name=cfg.MODEL_NAME,
        input_size=cfg.INPUT_SIZE,
        num_samples=cfg.NUM_INFERENCES,
        processor_flops_per_second=cfg.PROCESSOR_FLOPS_PER_SECOND,
        processor_max_power=cfg.PROCESSOR_MAX_POWER
    )


    # Calculate energy for each component
    transmission_calculation = transmission.calculate_energy(cfg.DATA_SIZE*cfg.FLOAT_PRECISION)
    transmission_energy = transmission_calculation['total_energy']
    storage_calculation = storage.calculate_energy(cfg.DATA_SIZE*cfg.FLOAT_PRECISION)
    storage_energy = storage_calculation['total_energy']
    
    preprocessing_calculation = preprocessing.calculate_energy(cfg.DATA_SIZE*cfg.FLOAT_PRECISION)
    preprocessing_energy = preprocessing_calculation['total_energy']

    training_energy_calculation = training.calculate_energy()
    training_energy = training_energy_calculation['training_energy']
    evaluation_energy = training_energy_calculation['evaluation_energy']

    
    inference_energy = inference.calculate_energy()
    

    
    # Sum up total energy consumption
    total_energy = (
        transmission_energy +
        storage_energy +
        preprocessing_energy +
        training_energy +
        evaluation_energy+
        inference_energy 
)
    
    return {
        'transmission': transmission_energy,
        'storage': storage_energy,
        'preprocessing': preprocessing_energy,
        'training': training_energy,
        'evaluation': evaluation_energy,
        'inference': inference_energy,
        'total': total_energy
    }

if __name__ == "__main__":
    # Execute energy calculations
    energy_results = calculate_total_energy()
    
    # Print results
    print("\nEnergy Consumption Results (in Joules):")
    print("-" * 40)
    for component, energy in energy_results.items():
        print(f"{component.capitalize()}: {energy:.2f} J")




