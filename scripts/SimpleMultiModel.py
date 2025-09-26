import sys
import os
# Add the parent directory to Python path to access calculators and configs modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calculators.Transmission import Transmission
from calculators.DataPreprocessing import DataPreprocessing
from calculators.Inference import Inference
from calculators.Training import Training
from calculators.ModelFLOPS import KANCalculator, TransformerCalculator
import calculators.ToyModels as toy_models
from configs import SimpleMultiModelConfig as cfg
########################CONFIG##################################



def calculate_total_energy():
    full_dict = {}
    for model_name in cfg.MODEL_NAMES:
        transmission_dict = {}
        # Initialize calculators
        for k, v in cfg.TRANSMISSON_HOPS.items():
            transmission = Transmission(
                failure_rate=v["FAILURE_RATE"],
                application=v["APPLICATION_PROTOCOLS"],
                presentation=v["PRESENTATION_PROTOCOLS"],
                session=v["SESSION_PROTOCOLS"],
                transport=v["TRANSPORT_PROTOCOLS"],
                network=v["NETWORK_PROTOCOLS"],
                datalink=v["DATALINK_PROTOCOLS"],
                physical=v["PHYSICAL_PROTOCOLS"]
            )
            transmission_dict[k] = transmission

        preprocessing = DataPreprocessing(
            preprocessing_type=cfg.PREPROCESSING_TYPE,
            processor_flops_per_second=cfg.DP_PROCESSOR_FLOPS_PER_SECOND,
            processor_max_power=cfg.DP_PROCESSOR_MAX_POWER,
            time_steps=cfg.SAMPLE_SIZE,  # only needed for GADF
        )
        if model_name == "KAN":
            calculator = KANCalculator(
                num_layers=cfg.NUM_LAYERS,
                grid_size=cfg.GRID_SIZE,
                num_classes=cfg.NUM_CLASSES,
                din=cfg.DIN,
                dout=cfg.DOUT,
                num_samples=cfg.SAMPLE_SIZE  # for time series

            )
        elif model_name == "SimpleTransformer":
            calculator = TransformerCalculator(
                context_length=cfg.CONTEXT_LENGTH,
                embedding_size=cfg.EMBEDDING_SIZE,
                num_heads=cfg.NUM_HEADS,
                num_decoder_blocks=cfg.NUM_DECODER_BLOCKS,
                feed_forward_size=cfg.FEED_FORWARD_SIZE,
                vocab_size=cfg.VOCAB_SIZE
            )

        else:
            calculator = None

        if model_name == "SimpleMLP":
            model = toy_models.SimpleMLP(input_size=cfg.INPUT_SIZE[1])
            training = Training(
                model_name=model,
                num_epochs=cfg.NUM_EPOCHS,
                batch_size=cfg.BATCH_SIZE,
                processor_flops_per_second=cfg.TR_PROCESSOR_FLOPS_PER_SECOND,
                processor_max_power=cfg.TR_PROCESSOR_MAX_POWER,
                num_samples=cfg.NUM_SAMPLES,
                input_size=cfg.INPUT_SIZE,
                evaluation_strategy=cfg.EVALUATION_STRATEGY,
                k_folds=cfg.K_FOLDS,
                split_ratio=cfg.SPLIT_RATIO,
                calculator=calculator
            )
            inference = Inference(
                model_name=model,
                input_size=cfg.INPUT_SIZE,
                num_samples=cfg.NUM_INFERENCES,
                processor_flops_per_second=cfg.INF_PROCESSOR_FLOPS_PER_SECOND,
                processor_max_power=cfg.INF_PROCESSOR_MAX_POWER,
                calculator=calculator
            )
        elif model_name == "SimpleCNN":
            model = toy_models.SimpleCNN()

            training = Training(
                model_name=model,
                num_epochs=cfg.NUM_EPOCHS,
                batch_size=cfg.BATCH_SIZE,
                processor_flops_per_second=cfg.TR_PROCESSOR_FLOPS_PER_SECOND,
                processor_max_power=cfg.TR_PROCESSOR_MAX_POWER,
                num_samples=cfg.NUM_SAMPLES,
                input_size=cfg.INPUT_SIZE_CNN,
                evaluation_strategy=cfg.EVALUATION_STRATEGY,
                k_folds=cfg.K_FOLDS,
                split_ratio=cfg.SPLIT_RATIO,
                calculator=calculator
            )

            inference = Inference(
                model_name=model,
                input_size=cfg.INPUT_SIZE_CNN,
                num_samples=cfg.NUM_INFERENCES,
                processor_flops_per_second=cfg.INF_PROCESSOR_FLOPS_PER_SECOND,
                processor_max_power=cfg.INF_PROCESSOR_MAX_POWER,
                calculator=calculator
            )
        else:
            training = Training(
                model_name=model_name,
                num_epochs=cfg.NUM_EPOCHS,
                batch_size=cfg.BATCH_SIZE,
                processor_flops_per_second=cfg.TR_PROCESSOR_FLOPS_PER_SECOND,
                processor_max_power=cfg.TR_PROCESSOR_MAX_POWER,
                num_samples=cfg.NUM_SAMPLES,
                input_size=cfg.INPUT_SIZE,
                evaluation_strategy=cfg.EVALUATION_STRATEGY,
                k_folds=cfg.K_FOLDS,
                split_ratio=cfg.SPLIT_RATIO,
                calculator=calculator
            )

            inference = Inference(
                model_name=model_name,
                input_size=cfg.INPUT_SIZE,
                num_samples=cfg.NUM_INFERENCES,
                processor_flops_per_second=cfg.INF_PROCESSOR_FLOPS_PER_SECOND,
                processor_max_power=cfg.INF_PROCESSOR_MAX_POWER,
                calculator=calculator
            )

        # Calculate energy for each component
        transmission_calculation = transmission.calculate_energy(cfg.NUM_SAMPLES * cfg.FLOAT_PRECISION * cfg.SAMPLE_SIZE)
        transmission_energy = transmission_calculation['total_energy']
        
        for hop in transmission_dict:
                transmission = transmission_dict[hop]
                transmission_calculation = transmission.calculate_energy(cfg.NUM_SAMPLES * cfg.FLOAT_PRECISION * cfg.SAMPLE_SIZE)
                transmission_energy += transmission_calculation['total_energy']




        preprocessing_calculation = preprocessing.calculate_energy(cfg.NUM_SAMPLES, cfg.SAMPLE_SIZE)
        preprocessing_energy = preprocessing_calculation['total_energy']

        training_energy_calculation = training.calculate_energy()
        training_energy = training_energy_calculation['training_energy']
        evaluation_energy = training_energy_calculation['evaluation_energy']

        if cfg.EVALUATION_STRATEGY == 'train_test_split':
            pass
        elif cfg.EVALUATION_STRATEGY == 'cross_validation':
            pass
        else:
            raise ValueError(f"Unsupported evaluation strategy: {cfg.EVALUATION_STRATEGY}")

        inference_energy = inference.calculate_energy()

        inference_transmission = transmission.calculate_energy(cfg.NUM_INFERENCES * cfg.FLOAT_PRECISION * cfg.SAMPLE_SIZE)
        inference_transmission_energy = inference_transmission["total_energy"]

        inference_preprocessing = preprocessing.calculate_energy(cfg.NUM_INFERENCES, cfg.SAMPLE_SIZE)
        inference_preprocessing_energy = inference_preprocessing["total_energy"]

        inference_process = inference_energy + inference_transmission_energy + inference_preprocessing_energy
        # Sum up total energy consumption
        total_energy = (
                transmission_energy +

                preprocessing_energy +
                training_energy +
                evaluation_energy +
                inference_process
        )

        total_energy = total_energy * (1 + cfg.VIRTUALIZATION_OVERHEAD)  # overhead due to virtualization
        curr_dict =  {
            'transmission': transmission_energy,
            'preprocessing': preprocessing_energy,
            'training': training_energy,
            'evaluation': evaluation_energy,
            'inference': inference_energy,
            'inference_process': inference_process,
            'total': total_energy,
            'Ed bits': cfg.FLOAT_PRECISION * cfg.SAMPLE_SIZE * cfg.NUM_SAMPLES,
            'inf_proc_bits': cfg.FLOAT_PRECISION * cfg.SAMPLE_SIZE * cfg.NUM_INFERENCES,
            'total_bits': cfg.FLOAT_PRECISION * cfg.SAMPLE_SIZE * cfg.NUM_SAMPLES + cfg.FLOAT_PRECISION * cfg.SAMPLE_SIZE * cfg.NUM_INFERENCES
        }
        full_dict[model_name] = curr_dict

    return full_dict

if __name__ == "__main__":


    print("""

    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    ░░█████░░░████░░░░███████░██░░░░░░░░░░░░░░░░░░░░░░░
    ░░██░░░░░██░░░██░░██░░░██░██░░░░░░░░░░░░░░░░░░░░░░░
    ░░████░░░██░░░░░░░███████░██░░░░░░░░░░░░░░░░░░░░░░░
    ░░██░░░░░██░░░██░░██░░░██░██░░░░░░░░░░░░░░░░░░░░░░░
    ░░█████░░░████░░░░██░░░██░█████░░░░░░░░░░░░░░░░░░░░
    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
          
    Authors : Shih-Kai Chou, Jernej Hribar, Vid Hanžel, Mihael Mohorčič, Carolina Fortuna
    Jožef Stefan Institute, Ljubljana, Slovenia
    DOI: https://doi.org/10.48550/arXiv.2408.00540

    """)
    # Execute energy calculations
    energy_results = calculate_total_energy()

    # Print results
    print("\nEnergy Consumption Results (in Joules):")
    print("-" * 40)

    for model_name, energy_dict in energy_results.items():
        print(f"\nModel: {model_name}")
        print("-" * 40)
        for component, energy in energy_dict.items():
            if component.capitalize() == "Total_bits" or component.capitalize() == "Inf_proc_bits" or component.capitalize() == "Ed bits":
                continue
            print(f"{component.capitalize()}: {energy:.4f} J ({energy / energy_dict['total'] * 100:.4f} %)")

        eCal = energy_dict['total'] / energy_dict['total_bits']
        print(f"eCAL: {eCal} J/bit")
