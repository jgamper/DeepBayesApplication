import torch
import numpy as np
from torch.autograd import Variable
from CapsuleNet.capsulenet import CapsuleNet
from scipy.stats import entropy

def naturality_score(test_loader, batch_size, gpu_id, weights_path, model=None):
    """
    Computes naturality score for MNIST, as Inception score proposed in [1],
    but modified according to [2] and using CapsuleNet trained on MNIST

    Reference below
    [1] Salimans, Tim, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen, and Xi Chen.
    “Improved Techniques for Training GANs.” In Advances in Neural Information
    Processing Systems 29, edited by D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon,
    and R. Garnett, 2234–2242. Curran Associates, Inc., 2016.
    http://papers.nips.cc/paper/6125-improved-techniques-for-training-gans.pdf.
    [2] Barratt, Shane, and Rishi Sharma. “A Note on the Inception Score.” ArXiv:1801.01973 [Cs, Stat], January 6, 2018. http://arxiv.org/abs/1801.01973.
    """
    # Initialise the CapsuleNet
    capsNet = CapsuleNet(input_size=[1, 28, 28], classes=10, routings=3)
    capsNet = capsNet.cuda(gpu_id)
    capsNet.load_state_dict(torch.load(weights_path))
    capsNet = capsNet.eval()

    # Freeze layers
    for p in capsNet.parameters():
        p.requires_grad = False

    # Proceed to compute the score
    N = test_loader.dataset.__len__()
    preds = np.zeros((N, 10))

    if model != None:
        for i, batch in enumerate(test_loader, 0):
            x, lbl = batch
            x = Variable(x.view(-1, 784).cuda(0))
            x = model.forward(x)
            batch_size_i = x.size()[0]
            x = x.view(batch_size_i, 1, 28, 28)
            preds[i*batch_size:i*batch_size + batch_size_i] = capsNet.forward(x)[0].cpu().data.numpy()
    else:
        for i, batch in enumerate(test_loader, 0):
            x, lbl = batch
            x = Variable(x.cuda(0), requires_grad=False)
            batch_size_i = x.size()[0]

            preds[i*batch_size:i*batch_size + batch_size_i] = capsNet.forward(x)[0].cpu().data.numpy()

    scores = []
    marginal_y = np.mean(preds, axis=0)
    for i in range(preds.shape[0]):
        conditional_y = preds[i,:]
        scores.append(entropy(conditional_y, marginal_y))

    return np.mean(scores), np.std(scores)
