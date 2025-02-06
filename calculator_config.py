######################################## Transmission ########################################
# select the protocol for each layer
PHYSICAL_PROTOCOLS = "Generic_physical" # set the protocol for the physical layer
DATALINK_PROTOCOLS = "Generic_datalink" # set the protocol for the datalink layer
NETWORK_PROTOCOLS = "Generic_network" # set the protocol for the network layer
TRANSPORT_PROTOCOLS = "Generic_transport" # set the protocol for the transport layer
SESSION_PROTOCOLS = "Generic_session" # set the protocol for the session layer
PRESENTATION_PROTOCOLS = "Generic_presentation" # set the protocol for the presentation layer
APPLICATION_PROTOCOLS = "Generic_application" # set the protocol for the application layer

FAILURE_RATE = 0.0 # Select transmission failure rate causing retransmission given as a float between 0 and 1


######################################## Data Preprocessing ########################################
PREPROCESSING_TYPE = "normalization" # set the preprocessing to apply to the data options: normalization, min_max_scaling, GADF

######################################## Storage ########################################
STORAGE_TYPE = "HDD" # set the storage type to use options: HDD, SSD
RAID_LEVEL = "NO_RAID" # set the RAID level to use options: RAID0, RAID1, RAID5, RAID6, NO_RAID
NUM_DISKS = 1 # set the number of disks to use only relevant if RAID_LEVEL is not NO_RAID


######################################## Training ########################################
MODEL_NAME ="KAN" # set which model to use examples: KAN, resnet18, baichuan-inc/Baichuan-13B-Chat
NUM_EPOCHS = 50
BATCH_SIZE = 32
# INPUT_SIZE = (1, 3, 224, 224) # 4d input for resnet # (1,128) # batch, max_seq_length for llm
INPUT_SIZE = (1, 10) # 

EVALUATION_STRATEGY = "cross_validation" # set the evaluation strategy options: cross_validation, train_test_split
K_FOLDS = 5 # Only used if EVALUATION_STRATEGY is cross_validation
SPLIT_RATIO = 0.8 # Only used if EVALUATION_STRATEGY is train_test_split


#KAN specfic parameters 
NUM_LAYERS = 3
GRID_SIZE = 10
NUM_CLASSES = 2 
DIN = 10
DOUT = 2

# Transformer specific parameters
CONTEXT_LENGTH = 10  
EMBEDDING_SIZE = 16  
NUM_HEADS = 2       
NUM_DECODER_BLOCKS = 3  
FEED_FORWARD_SIZE = 32  

######################################## Inference ########################################
NUM_INFERENCES = 10000 # number of inferences to run


# GENERAL CONFIG
NUM_SAMPLES = 1000 # number of samples that are used to calculate the energy consumption
SAMPLE_SIZE = 10 # size of a single sample e.g. number of timesteps in a timeseries or number of pixels in an image
FLOAT_PRECISION = 64 # number of bits used to represent a floating point number
PROCESSOR_FLOPS_PER_SECOND = 1e12 # theoretical maximum number of floating point operations per second for the processor
PROCESSOR_MAX_POWER = 100 # maximum power consumption of the processor in Watts
