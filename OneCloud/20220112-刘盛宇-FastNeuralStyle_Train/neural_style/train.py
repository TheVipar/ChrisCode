import os
import sys
import time
import random

import cv2
import numpy as np
import oneflow as flow
from oneflow.optim import Adam
import flowvision

import utils
from utils import load_image, recover_image, normalize_batch, load_image_eval
from transformer_net import TransformerNet
from vgg_features import VGG_WITH_FEATURES
from config import *


def check_paths():
    try:
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        if CHECKPOINT_DIR is not None and not (os.path.exists(CHECKPOINT_DIR)):
            os.makedirs(CHECKPOINT_DIR)
    except OSError as e:
        print(e)
        sys.exit(1)


def train():
    device = "cuda"
    np.random.seed(SEED)
    # load path of train images
    train_images = os.listdir(DATASET)
    train_images = [image for image in train_images if not image.endswith("txt")]
    random.shuffle(train_images)
    images_num = len(train_images)
    print("dataset size: %d" % images_num)
    # Initialize transforemer net, optimizer, and loss function
    transformer = TransformerNet().to("cuda")

    optimizer = Adam(transformer.parameters(), LR)
    mse_loss = flow.nn.MSELoss()

    if LOAD_CHECKPOINT_DIR is not None:
        state_dict = flow.load(LOAD_CHECKPOINT_DIR)
        transformer.load_state_dict(state_dict)
        print("successfully load checkpoint from " + LOAD_CHECKPOINT_DIR)

    if MODEL == "vgg19":
        vgg = flowvision.models.vgg19(pretrained=True)
    else:
        vgg = flowvision.models.vgg16(pretrained=True)
    vgg = VGG_WITH_FEATURES(vgg.features, requires_grad=False)
    vgg.to("cuda")

    style_image = utils.load_image(STYLE_IMAGE)
    style_image_recover = recover_image(style_image)
    features_style = vgg(utils.normalize_batch(flow.Tensor(style_image).to("cuda")))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for e in range(EPOCHS):
        transformer.train()
        agg_content_loss = 0.0
        agg_style_loss = 0.0
        count = 0
        for i in range(images_num):
            image = load_image("%s/%s" % (DATASET, train_images[i]))
            n_batch = BATCH
            count += n_batch

            x_gpu = flow.tensor(image, requires_grad=True).to("cuda")
            y_origin = transformer(x_gpu)

            x_gpu = utils.normalize_batch(x_gpu)
            y = utils.normalize_batch(y_origin)

            features_x = vgg(x_gpu)
            features_y = vgg(y)
            content_loss = CONTENT_WEIGHT * mse_loss(
                features_y.relu2_2, features_x.relu2_2
            )
            style_loss = 0.0
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= STYLE_WEIGHT

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            agg_content_loss += content_loss.numpy()
            agg_style_loss += style_loss.numpy()
            if (i + 1) % LOG_INTERVAL == 0:
                if STYLE_DIR is not None:
                    y_recover = recover_image(y_origin.numpy())
                    image_recover = recover_image(image)
                    result = np.concatenate(
                        (style_image_recover, image_recover), axis=1
                    )
                    result = np.concatenate((result, y_recover), axis=1)
                    cv2.imwrite(STYLE_DIR + str(i + 1) + ".jpg", result)
                    print(STYLE_DIR + str(i + 1) + ".jpg" + " saved")
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(),
                    e + 1,
                    count,
                    images_num,
                    agg_content_loss / (i + 1),
                    agg_style_loss / (i + 1),
                    (agg_content_loss + agg_style_loss) / (i + 1),
                )
                print(mesg)

            if CHECKPOINT_DIR is not None and (i + 1) % CHECKPOINT_INTERVAL == 0:
                transformer.eval()
                ckpt_model_filename = (
                    "CW_"
                    + str(int(CONTENT_WEIGHT))
                    + "_lr_"
                    + str(LR)
                    + "ckpt_epoch"
                    + str(e)
                    + "_"
                    + str(i + 1)
                )
                ckpt_model_path = os.path.join(CHECKPOINT_DIR, ckpt_model_filename)
                flow.save(transformer.state_dict(), ckpt_model_path)
                transformer.train()

    # save model
    transformer.eval()
    save_model_filename = (
        "CW_"
        + str(CONTENT_WEIGHT)
        + "_lr_"
        + str(LR)
        + "sketch_epoch_"
        + str(EPOCHS)
        + "_"
        + str(time.ctime()).replace(" ", "_")
        + "_"
        + str(CONTENT_WEIGHT)
        + "_"
        + str(STYLE_WEIGHT)
    )
    save_model_path = os.path.join(SAVE_DIR, save_model_filename)
    flow.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def main():
    check_paths()
    train()


if __name__ == "__main__":
    main()
