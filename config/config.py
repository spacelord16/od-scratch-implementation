class Config:
    # Data
    DATA_PAth = "od-scratch\dataset"
    MODEL_SAVE_PATH = "od-scratch\model_save"

    # Hyper Parameters (can be tuned every time)
    LEARNING_RATE = 0.001
    BATCH_sIZE = 16
    EPOCHS = 100

    # Model config
    IMG_HEIGHT = 416
    IMG_WIDTH = 416
    CHANNELS = 3

    # Rest misc settings
    SEED = 42
