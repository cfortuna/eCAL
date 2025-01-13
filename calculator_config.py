######################################## Transmission ########################################
# select the protocol for each layer
PHYSICAL_PROTOCOLS = "WIFI_PHY"
DATALINK_PROTOCOLS = "ETHERNET"
NETWORK_PROTOCOLS = "IPv4"
TRANSPORT_PROTOCOLS = "TCP"
SESSION_PROTOCOLS = "RPC"
PRESENTATION_PROTOCOLS = "TLS"
APPLICATION_PROTOCOLS = "HTTP"
# Select transmission failure rate causing retransmission given as a float between 0 and 1
FAILURE_RATE = 0.0


######################################## Data Preprocessing ########################################
PREPROCESSING_TYPE = "normalization"

######################################## Storage ########################################
STORAGE_TYPE = "HDD"
RAID_LEVEL = "NO_RAID"
NUM_DISKS = 1


######################################## Training ########################################
MODEL_NAME = "resnet18"
NUM_EPOCHS = 10
BATCH_SIZE = 32
INPUT_SIZE = (1, 3, 224, 224)# 4d input for resnet
EVALUATION_STRATEGY = "cross_validation" # train_test_split or cross_validation
K_FOLDS = 5 # Only used if EVALUATION_STRATEGY is cross_validation
SPLIT_RATIO = 0.8 # Only used if EVALUATION_STRATEGY is train_test_split


######################################## Inference ########################################
NUM_INFERENCES = 10000


# GENERAL CONFIG
DATA_SIZE = 1000
FLOAT_PRECISION = 64
PROCESSOR_FLOPS_PER_SECOND = 1e12
PROCESSOR_MAX_POWER = 100
