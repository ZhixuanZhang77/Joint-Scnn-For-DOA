class Settings:
    def __init__(self):
        self.training = config()

class config:
    def __init__(self):
        self.device = 'cuda:0'#'cpu'#

SETTINGS = Settings()
