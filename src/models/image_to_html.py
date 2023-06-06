from torch import nn

class image_model(nn.Module):
    '''
    CNN model for processing image files of website mock-up screenshots.
    '''
    def __init__(self):
        super().__init__()

    def forward(x):
        pass


class language_model(nn.Module):
    '''
    LSTM model for processing html context files. The context is
    preprocessed into one-hot encoded tokens.
    '''
    def __init__(self):
        super().__init__()

    def forward(x):
        pass

class decode_model(nn.Module):
    '''
    LSTM model for processing concatenated output from previous image 
    and language models.
    '''
    def __init__(self):
        super().__init__()

    def forward(x):
        pass

class image_to_html(nn.Module):
    '''
    Full model for training on image and html context files.
    Processes image files through a CNN and html files through an LSTM.
    The results of these models are concatenated and fed into a second 
    LSTM model that decodes them back into tokens one at a time.
    '''
    def __init__(self):
        super().__init__()

        # Gets repeated for as many tokens there are in the context
        self.image_model = nn.Sequential(
            # layer 1 - 32
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0), nn.ReLU6,
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0), nn.ReLU6,
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            # layer 2 - 64
            nn.Conv2d(x, 64, kernel_size=3, padding=0, stride=1), nn.ReLU6,
            nn.Conv2d(x, 64, kernel_size=3, padding=0, stride=1), nn.ReLU6,
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            # layer 3 - 128
            nn.Conv2d(x, 128, kernel_size=3, padding=0, stride=1), nn.ReLU6,
            nn.Conv2d(x, 128, kernel_size=3, padding=0, stride=1), nn.ReLU6,
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            # flatten and reshape
            nn.Flatten(),
            nn.Linear(1024), nn.ReLU6,
            nn.Dropout(0.3),
            nn.Linear(1024), nn.ReLU6,
            nn.Dropout(0.3)
        )

        self.language_model = nn.Sequential(
            nn.LSTM() #TODO: Here
        )




    def forward(x):
        pass


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)