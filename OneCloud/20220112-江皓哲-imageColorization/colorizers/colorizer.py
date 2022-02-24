import argparse
import os

from skimage import io
from colorizers import *
from PIL import Image

class Colorizer:

    def __init__(self, model_path, use_gpu=False):
        # Load model
        self.model = SIGGRAPHGenerator().cuda() if use_gpu else SIGGRAPHGenerator()
        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
        self.use_gpu = use_gpu

    def colorize(self, img_nparr):
        self.model.eval()
        # Load and process image
        # img = np.asarray(Image.open(img_path).convert('RGB'))  # ndarray
        # img_name = os.path.basename(img_path)
        # img = io.imread(fname=img_path, as_gray=True)
        
        print(img_nparr.shape)
        h = img_nparr.shape[0]
        w = img_nparr.shape[1]

        L_orig, L_256 = rgb2L_tensor(img_orig=img_nparr, HW=(256, 256), resample=3)
        if (self.use_gpu):
            L_256 = L_256.cuda()

        img_colorized = Lab2rgb_numpy(L_orig=L_orig, out_ab=self.model(L_256).cpu())
        assert img_colorized.shape == (h, w, 3)

        # colorized_img_name = "colorized-" + img_name
        # io.imsave(colorized_img_name,img_colorized)

        
        return img_colorized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to colorize a photo")
    parser.add_argument('image', help="Path to the image. RGB one will be converted to grayscale")
    parser.add_argument('model', help="Path a *.pth model")
    args = parser.parse_args()

    model = Colorizer(model_path=args.model)
    model.colorize(img_path=args.image)
