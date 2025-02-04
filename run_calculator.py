from calculators.Storage import Storage
from eCAL.calculators.TransmissionSimple import TransmissionSimple
from calculators.DataPreprocessing import DataPreprocessing
from calculators.Inference import Inference
from calculators.Training import Training
from eCAL.calculators.ModelFLOPS import KANCalculator
import calculator_config as cfg

def calculate_total_energy():
    # Initialize calculators
    transmission = TransmissionSimple(
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
        time_steps=cfg.SAMPLE_SIZE, # only needed for GADF
    )
    if cfg.MODEL_NAME == "KAN":
        calculator = KANCalculator(
            num_layers=cfg.NUM_LAYERS,
            grid_size=cfg.GRID_SIZE,
            num_classes=cfg.NUM_CLASSES,
            din=cfg.DIN,
            dout=cfg.DOUT,
            num_samples=cfg.SAMPLE_SIZE # for time series

        )
    else:
        calculator = None


    training = Training(
        model_name=cfg.MODEL_NAME,
        num_epochs=cfg.NUM_EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        processor_flops_per_second=cfg.PROCESSOR_FLOPS_PER_SECOND,
        processor_max_power=cfg.PROCESSOR_MAX_POWER,
        num_samples=cfg.NUM_SAMPLES,
        input_size=cfg.INPUT_SIZE,
        evaluation_strategy=cfg.EVALUATION_STRATEGY,
        k_folds=cfg.K_FOLDS,
        split_ratio=cfg.SPLIT_RATIO,
        calculator=calculator
    )
    
    
    inference = Inference(
        model_name=cfg.MODEL_NAME,
        input_size=cfg.INPUT_SIZE,
        num_samples=cfg.NUM_INFERENCES,
        processor_flops_per_second=cfg.PROCESSOR_FLOPS_PER_SECOND,
        processor_max_power=cfg.PROCESSOR_MAX_POWER,
        calculator=calculator
    )


    # Calculate energy for each component
    transmission_calculation = transmission.calculate_energy(cfg.NUM_SAMPLES*cfg.FLOAT_PRECISION*cfg.SAMPLE_SIZE)
    transmission_energy = transmission_calculation['total_energy']
    transmission_bits = transmission_calculation['total_bits']

    storage_calculation = storage.calculate_energy(cfg.NUM_SAMPLES*cfg.FLOAT_PRECISION*cfg.SAMPLE_SIZE)
    storage_energy = storage_calculation['total_energy']
    storage_bits = storage_calculation['details']['raw_storage_bits']
    
    preprocessing_calculation = preprocessing.calculate_energy(cfg.NUM_SAMPLES, cfg.SAMPLE_SIZE)
    preprocessing_energy = preprocessing_calculation['total_energy']
    preprocessing_bits = preprocessing_calculation['total_bits']

    training_energy_calculation = training.calculate_energy()
    training_energy = training_energy_calculation['training_energy']
    evaluation_energy = training_energy_calculation['evaluation_energy']

    if cfg.EVALUATION_STRATEGY == 'train_test_split':
        training_bits = cfg.NUM_SAMPLES * cfg.FLOAT_PRECISION * cfg.SAMPLE_SIZE * cfg.SPLIT_RATIO
        evaluation_bits = cfg.NUM_SAMPLES * cfg.FLOAT_PRECISION * cfg.SAMPLE_SIZE * (1 - cfg.SPLIT_RATIO)
    elif cfg.EVALUATION_STRATEGY == 'cross_validation':
        training_bits = cfg.NUM_SAMPLES * cfg.FLOAT_PRECISION * cfg.SAMPLE_SIZE * (1 - 1 / cfg.K_FOLDS)
        evaluation_bits = cfg.NUM_SAMPLES * cfg.FLOAT_PRECISION * cfg.SAMPLE_SIZE * (1 / cfg.K_FOLDS)
    else:
        raise ValueError(f"Unsupported evaluation strategy: {cfg.EVALUATION_STRATEGY}")

    
    inference_energy = inference.calculate_energy()
    inference_bits = cfg.NUM_INFERENCES * cfg.FLOAT_PRECISION * cfg.SAMPLE_SIZE
    
    inference_transmission = transmission.calculate_energy(cfg.NUM_INFERENCES*cfg.FLOAT_PRECISION*cfg.SAMPLE_SIZE)
    inference_transmission_energy = inference_transmission["total_energy"]
    inference_transmission_bits = inference_transmission["total_bits"]


    inference_storage = storage.calculate_energy(cfg.NUM_INFERENCES*cfg.FLOAT_PRECISION*cfg.SAMPLE_SIZE)
    inference_storage_energy = inference_storage["total_energy"]
    inference_storage_bits = inference_storage["details"]["raw_storage_bits"]

    inference_preprocessing = preprocessing.calculate_energy(cfg.NUM_INFERENCES, cfg.SAMPLE_SIZE)
    inference_preprocessing_energy = inference_preprocessing["total_energy"]
    inference_preprocessing_bits = inference_preprocessing["total_bits"]
    

    inference_process = inference_energy + inference_transmission_energy + inference_storage_energy + inference_preprocessing_energy
    # Sum up total energy consumption
    total_energy = (
        transmission_energy +
        storage_energy +
        preprocessing_energy +
        training_energy +
        evaluation_energy+
        inference_process
)
    
    return {
        'transmission': transmission_energy,
        'storage': storage_energy,
        'preprocessing': preprocessing_energy,
        'training': training_energy,
        'evaluation': evaluation_energy,
        'inference': inference_energy,
        'inference_process' : inference_process,
        'total': total_energy,
        'Ed bits' : transmission_bits + storage_bits + preprocessing_bits + training_bits + evaluation_bits,
        'inf_proc_bits' : inference_transmission_bits + inference_storage_bits + inference_preprocessing_bits + inference_bits,
        'total_bits' : transmission_bits + storage_bits + preprocessing_bits + training_bits + evaluation_bits + (inference_transmission_bits + inference_storage_bits + inference_preprocessing_bits + inference_bits)
    }

if __name__ == "__main__":
    # Execute energy calculations
    energy_results = calculate_total_energy()
    
    # Print results
    print("\nEnergy Consumption Results (in Joules):")
    print("-" * 40)
    for component, energy in energy_results.items():
        if component.capitalize() == "Total_bits":
            continue
        print(f"{component.capitalize()}: {energy:.4f} J ({energy/energy_results['total']*100:.4f} %)")


    
    eCal = energy_results['total']/ energy_results['total_bits']
    print(f"eCAL: {eCal} J/bit")



