class SplitInTwo:
    def __init__(self, transform): 
        self.transform = transform

    def __call__(self, img):
        return [self.transform(img), self.transform(img)]