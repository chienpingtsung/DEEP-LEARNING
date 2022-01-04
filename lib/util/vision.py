import math

from PIL import Image
from torchvision.transforms import functional


class Tile:
    """
    Process image as description from Figure 2 of
    https://arxiv.org/pdf/1505.04597.pdf
    """

    def __init__(self, image_res, label_res):
        """As description from paper, mask after forward propagation should be central region of original image,
        because convolutions cause losing verge information, so only the centre could represent complete information.
        :param image_res: Target image resolution after Tile operation.
        :param mask_res: Target mask resolution after Tile operation.
        """
        assert (image_res - label_res) % 2 == 0, "Gap between image and label_res should be dividable by 2."

        self.image_res = image_res
        self.label_res = label_res

    def __call__(self, image, *args):
        """Call for functional using.
        :param image: PIL Image.
        :param mask: (optional) Original mask (H, W) for splitting. None for prediction purpose.
        """
        width, height = image.size

        n_row = math.ceil(height / self.label_res)
        n_col = math.ceil(width / self.label_res)

        padded_height = n_row * self.label_res
        padded_width = n_col * self.label_res

        top = (padded_height - height) // 2
        bot = math.ceil((padded_height - height) / 2)
        lef = (padded_width - width) // 2
        rig = math.ceil((padded_width - width) / 2)

        cornicione = (self.image_res - self.label_res) // 2

        image = functional.pad(image, [
            lef + cornicione,  # left
            top + cornicione,  # top
            rig + cornicione,  # right
            bot + cornicione  # bottom
        ], padding_mode='reflect')
        images = [functional.crop(image, top, lef, self.image_res, self.image_res)
                  for top in range(0, padded_height, self.label_res)
                  for lef in range(0, padded_width, self.label_res)]

        rst = []
        for pic in args:
            pic = functional.pad(pic, [
                lef,
                top,
                rig,
                bot
            ], padding_mode='reflect')
            pics = [functional.crop(pic, top, lef, self.label_res, self.label_res)
                    for top in range(0, padded_height, self.label_res)
                    for lef in range(0, padded_width, self.label_res)]
            rst.append(pics)

        if rst:
            return images, *rst
        return images


class Detile:
    def __call__(self, pieces, orig_size, top, left):
        im = Image.new(pieces[0].mode, orig_size)
        width, height = orig_size
        hori_step, vert_step = pieces[0].size

        i = 0
        for y in range(-top, -top + height, vert_step):
            for x in range(-left, -left + width, hori_step):
                im.paste(pieces[i], (x, y))
                i += 1
        return im
