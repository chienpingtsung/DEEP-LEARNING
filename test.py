from pathlib import Path

import torch

from utils.evaluate import calc_confusion_matrix


def test(model, dataloader, device, save_to: Path = None):
    if save_to:
        save_to.mkdir(parents=True, exist_ok=True)

    model.eval()

    prec_TP = prec_TPFP = reca_TP = reca_TPFN = 0

    for image, label in dataloader:
        with torch.no_grad():
            image = image.to(device)

            output = model(image)
            output = torch.sigmoid(output)
            pred = torch.squeeze(output) > 0.5

            label = torch.squeeze(label).numpy().astype(bool)
            pred = pred.cpu().numpy()
            for la, pr in zip(label, pred):
                p_TP, p_TPFP, r_TP, r_TPFN = calc_confusion_matrix(la, pr, tolerance=2)
                prec_TP += p_TP
                prec_TPFP += p_TPFP
                reca_TP += r_TP
                reca_TPFN += r_TPFN

    infinitesimal = 1e-10
    prec = prec_TP / (prec_TPFP + infinitesimal)
    reca = reca_TP / (reca_TPFN + infinitesimal)
    f1 = (2 * prec * reca) / (prec + reca + infinitesimal)

    return prec, reca, f1
