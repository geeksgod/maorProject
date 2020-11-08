#!/usr/bin/env python

import os
import sys
import argparse
from tqdm import *

import numpy as np
import torch

from models import Text2Mel, SSRN
from hparams import HParams as hp
from audio import save_to_wav
from utils import get_last_checkpoint_file_name, load_checkpoint, save_to_png

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", required=True, choices=['npspeech'], help='dataset name')
args = parser.parse_args()

from datasets.np_speech import vocab, get_test_data

SENTENCES = [
    "स्वास्थ्य तथा जनसंख्या मन्त्रालयले आइतबार मृत्युको औपचारिक रुपमा पुष्टि गरेको हो ।",
    "पार्टीका दुई अध्यक्ष केपी शर्मा ओली र पुष्पकमल दाहालबीच पछिल्लो मतभेद सतहमा आएकै बेला महासचिव पौडेलले नेकपाको एकीकृत अस्तित्व गम्भीर संकटमा परेको बताएका हुन् ।",
    "तिहारपछि खुला खाना वितरण गरिने",
    "लकडाउनपछि खाना खुवाइरहेका अभियन्ताहरु र काठमाडौं महानगरपालिकाबीच भाइटीकाको भोलिपल्टदेखि निश्चित स्थान पहिचान गर्ने वा प्याकेटमा मात्र खाना वितरण गर्ने सहमति भएको हो ।",
    "प्रधानमन्त्री ओली र भारतीय सेनाध्यक्ष नरवणेबीच भेट",
    "दाहालले बोलाए धुम्बाराहीमा सचिवालयको अनौपचारिक बैठक",
    "पार्टी विवादमा राष्ट्रपतिको सक्रियता पदीय मर्यादाविपरीत",
    "अमेरिकामा एकैदिन एक लाख बीस हजारभन्दा बढीमा कोरोना संक्रमण ।",
    "प्रदेश प्रहरी गठनको माग गर्दै प्रदेश दुई सर्वोच्चमा",
]

torch.set_grad_enabled(False)

text2mel = Text2Mel(vocab).eval()
last_checkpoint_file_name = get_last_checkpoint_file_name(os.path.join(hp.logdir, '%s-text2mel' % args.dataset))
# last_checkpoint_file_name = 'logdir/%s-text2mel/step-020K.pth' % args.dataset
if last_checkpoint_file_name:
    print("loading text2mel checkpoint '%s'..." % last_checkpoint_file_name)
    load_checkpoint(last_checkpoint_file_name, text2mel, None)
else:
    print("text2mel not exits")
    sys.exit(1)

ssrn = SSRN().eval()
last_checkpoint_file_name = get_last_checkpoint_file_name(os.path.join(hp.logdir, '%s-ssrn' % args.dataset))
# last_checkpoint_file_name = 'logdir/%s-ssrn/step-005K.pth' % args.dataset
if last_checkpoint_file_name:
    print("loading ssrn checkpoint '%s'..." % last_checkpoint_file_name)
    load_checkpoint(last_checkpoint_file_name, ssrn, None)
else:
    print("ssrn not exits")
    sys.exit(1)

# synthetize by one by one because there is a batch processing bug!
for i in range(len(SENTENCES)):
    sentences = [SENTENCES[i]]

    max_N = len(SENTENCES[i])
    L = torch.from_numpy(get_test_data(sentences, max_N))
    zeros = torch.from_numpy(np.zeros((1, hp.n_mels, 1), np.float32))
    Y = zeros
    A = None

    for t in tqdm(range(hp.max_T)):
        _, Y_t, A = text2mel(L, Y, monotonic_attention=True)
        Y = torch.cat((zeros, Y_t), -1)
        _, attention = torch.max(A[0, :, -1], 0)
        attention = attention.item()
        if L[0, attention] == vocab.index('E'):  # EOS
            break

    _, Z = ssrn(Y)

    Y = Y.cpu().detach().numpy()
    A = A.cpu().detach().numpy()
    Z = Z.cpu().detach().numpy()

    save_to_png('samples/%d-att.png' % (i + 1), A[0, :, :])
    save_to_png('samples/%d-mel.png' % (i + 1), Y[0, :, :])
    save_to_png('samples/%d-mag.png' % (i + 1), Z[0, :, :])
    save_to_wav(Z[0, :, :].T, 'samples/%d-wav.wav' % (i + 1))
