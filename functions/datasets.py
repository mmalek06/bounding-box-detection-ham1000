from torchvision.datasets import CocoDetection


class CocoDetectionWithFilenames(CocoDetection):
    def __init__(self, root: str, annFile: str, transform=None):
        super().__init__(root, annFile, transform)

    def get_filename(self, idx: int) -> str:
        return self.coco.loadImgs(self.ids[idx])[0]["file_name"]
