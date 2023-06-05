# data generation
IFOS = ['H1', 'L1']
STRAIN_START = 1239134846
STRAIN_STOP = 1239140924
SAMPLE_RATE = 2048
BANDPASS_LOW = 30
BANDPASS_HIGH = 1024
GLITCH_SNR_BAR = 10
N_INJECTIONS = 25000
LOADED_DATA_SAMPLE_RATE = 16384
DATA_SEGMENT_LOAD_START = 0
DATA_SEGMENT_LOAD_STOP = 3600
INJECT_AT_END = False
EDGE_INJECT_SPACING = 1.2
INJECTION_SEGMENT_LENGTH = 4

# data sampling arguments
BBH_WINDOW_LEFT = -0.04
BBH_WINDOW_RIGHT = 0.01
BBH_AMPLITUDE_BAR = 5
BBH_N_SAMPLES = 5
SG_WINDOW_LEFT = -0.05
SG_WINDOW_RIGHT = 0.05
SG_AMPLITUDE_BAR = 5
SG_N_SAMPLES = 5
BKG_N_SAMPLES = 5
NUM_IFOS = 2
SEG_NUM_TIMESTEPS = 100
SEGMENT_OVERLAP = 5

# Glitch "generation"
SNR_THRESH = 5
GLITCH_START = 1238450550
GLITCH_STOP = 1253946511
Q_MIN = 3.3166
Q_MAX = 108
F_MIN = 32
CLUSTER_DT = 0.5
CHUNK_DURATION = 124
SEGMENT_DURATION = 64
OVERLAP = 4
MISTMATCH_MAX = 0.2
WINDOW = 2
CHANNEL = 'DCS-CALIB_STRAIN_CLEAN_C01'
FRAME_TYPE = 'HOFT_C01'
GLITCH_SAMPLE_RATE = 1024
STATE_FLAG = 'DCS-ANALYSIS_READY_C01:1'

# training
TEST_SPLIT = 0.9
BOTTLENECK = 15
FACTOR = 2
EPOCHS = 100
BATCH_SIZE = 100
LOSS = 'MAE'
OPTIMIZER = 'Adam'
VALIDATION_SPLIT = 0.15
TRAINING_VERBOSE = True


# timeslides
GW_EVENT_CLEANING_WINDOW = 5
TIMESLIDE_STEP = 0.5
TIMESLIDE_TOTAL_DURATION = 0.5*30*24*3600