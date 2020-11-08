
1. preprocess: `python dl_and_preprop_dataset.py --dataset=npspeech`
2. Train Text2Mel model: `python train-text2mel.py --dataset=npspeech`
3. Train SSRN model: `python train-ssrn.py --dataset=npspeech`
4. Synthesize sentences: `python synthesize.py --dataset=npspeech`