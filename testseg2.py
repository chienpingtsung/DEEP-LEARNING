from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from transforms.translate import ToTensor
from utils.evaluate import calc_confusion_matrix
from utils.vision import Tile, Detile


def test(model,
         image_root: Path,
         label1_root: Path,
         label2_root: Path,
         batch_size,
         device,
         threshold=0.5,
         save_to: Path = None):
    if save_to:
        save_to.mkdir(parents=True, exist_ok=True)

    tile = Tile(512, 512)
    detile = Detile()
    totensor = ToTensor()

    model.eval()

    seg_prec_TP = seg_prec_TPFP = seg_reca_TP = seg_reca_TPFN = 0
    edge_prec_TP = edge_prec_TPFP = edge_reca_TP = edge_reca_TPFN = 0
    merge_prec_TP = merge_prec_TPFP = merge_reca_TP = merge_reca_TPFN = 0

    for name in tqdm([p.name for p in image_root.glob('*.png')]):
        with torch.no_grad():
            image = Image.open(image_root.joinpath(name))
            label1 = Image.open(label1_root.joinpath(name))
            label2 = Image.open(label2_root.joinpath(name))

            images = tile(image)
            images = totensor(*images)
            images = torch.stack(images)

            seg_preds = []
            edge_preds = []
            merge_preds = []

            for i in range(0, len(images), batch_size):
                image = images[i:i + batch_size]

                image = image.to(device)

                seg_output, edge_output, merge_output = model(image)
                seg_output = torch.sigmoid(seg_output)
                edge_output = torch.sigmoid(edge_output)
                merge_output = torch.sigmoid(merge_output)
                seg_pred = torch.squeeze(seg_output) > threshold
                edge_pred = torch.squeeze(edge_output) > threshold
                merge_pred = torch.squeeze(merge_output) > threshold

                seg_pred = seg_pred.cpu().numpy().astype(np.uint8) * 255
                edge_pred = edge_pred.cpu().numpy().astype(np.uint8) * 255
                merge_pred = merge_pred.cpu().numpy().astype(np.uint8) * 255
                for i in range(len(seg_pred)):
                    seg_preds.append(Image.fromarray(seg_pred[i], 'L').convert('1'))
                    edge_preds.append(Image.fromarray(edge_pred[i], 'L').convert('1'))
                    merge_preds.append(Image.fromarray(merge_pred[i], 'L').convert('1'))

            seg_pred = detile(seg_preds, (1800, 900), (512 * 2 - 900) // 2, (512 * 4 - 1800) // 2)
            edge_pred = detile(edge_preds, (1800, 900), (512 * 2 - 900) // 2, (512 * 4 - 1800) // 2)
            merge_pred = detile(merge_preds, (1800, 900), (512 * 2 - 900) // 2, (512 * 4 - 1800) // 2)

            seg_pred = torch.squeeze(totensor(seg_pred)[0]).numpy().astype(bool)
            edge_pred = torch.squeeze(totensor(edge_pred)[0]).numpy().astype(bool)
            merge_pred = torch.squeeze(totensor(merge_pred)[0]).numpy().astype(bool)

            label1 = torch.squeeze(totensor(label1)[0]).numpy().astype(bool)
            label2 = torch.squeeze(totensor(label2)[0]).numpy().astype(bool)

            seg_p_TP, seg_p_TPFP, seg_r_TP, seg_r_TPFN = calc_confusion_matrix(label1, seg_pred, tolerance=2)
            edge_p_TP, edge_p_TPFP, edge_r_TP, edge_r_TPFN = calc_confusion_matrix(label2, edge_pred, tolerance=2)
            merge_p_TP, merge_p_TPFP, merge_r_TP, merge_r_TPFN = calc_confusion_matrix(label1, merge_pred, tolerance=2)

            seg_prec_TP += seg_p_TP
            seg_prec_TPFP += seg_p_TPFP
            seg_reca_TP += seg_r_TP
            seg_reca_TPFN += seg_r_TPFN
            edge_prec_TP += edge_p_TP
            edge_prec_TPFP += edge_p_TPFP
            edge_reca_TP += edge_r_TP
            edge_reca_TPFN += edge_r_TPFN
            merge_prec_TP += merge_p_TP
            merge_prec_TPFP += merge_p_TPFP
            merge_reca_TP += merge_r_TP
            merge_reca_TPFN += merge_r_TPFN

    infinitesimal = 1e-10
    seg_prec = seg_prec_TP / (seg_prec_TPFP + infinitesimal)
    seg_reca = seg_reca_TP / (seg_reca_TPFN + infinitesimal)
    seg_f1 = (2 * seg_prec * seg_reca) / (seg_prec + seg_reca + infinitesimal)
    edge_prec = edge_prec_TP / (edge_prec_TPFP + infinitesimal)
    edge_reca = edge_reca_TP / (edge_reca_TPFN + infinitesimal)
    edge_f1 = (2 * edge_prec * edge_reca) / (edge_prec + edge_reca + infinitesimal)
    merge_prec = merge_prec_TP / (merge_prec_TPFP + infinitesimal)
    merge_reca = merge_reca_TP / (merge_reca_TPFN + infinitesimal)
    merge_f1 = (2 * merge_prec * merge_reca) / (merge_prec + merge_reca + infinitesimal)

    return seg_prec, seg_reca, seg_f1, edge_prec, edge_reca, edge_f1, merge_prec, merge_reca, merge_f1
