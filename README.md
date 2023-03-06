# CodeTest
CodeTest for transfer MASc

Ruichen Yao

# Question
Try to re-implement the results of [FCRO](https://arxiv.org/pdf/2301.01481.pdf) with a new setting that you know the data can be biased on age, gender and color, but no sensitive attributes' labels are available

# Method
I try to use a unsupervised method to extract the sensitive features(age, bias, and gender), which is $Z_a$ in the paper. I use a $\beta$-VAE to extract the sensitive features. $\beta$-VAE is an improved version of VAE(Variational Autoencoder). In $\beta$-VAE, the $\beta$ value controls the weight of the KL divergence term, so the influence of the reconstruction loss and the KL divergence term can be balanced by adjusting the beta value. A higher $\beta$ value can promote more pronounced feature separation, but may lead to worse reconstruction performance.

Therefore, I select all images of helthy patients to train the $\beta$-VAE. In this way, I can control all input images to have the same properties in the target label. And the different features in these images are more likely to be sensitive features. At this time, I set a relatively large $\beta$ value(100 in this case) to extract more features containing sensitive information.

# Training method
0. Download the [CheXpert dataset](https://www.kaggle.com/datasets/mimsadiislam/chexpert). The label file is located at `./FCRO-Fair-Classification-Orthogonal-Representation/metadata/METADATA_reduced.xlsx`.

1. run `./unsupervised/betaVAE.ipynb` to generate the $\beta$-VAE for sensitive feature extraction. The weights are stored at `./unsupervised/model_weights_beta100_epoch10.pth`

2. run `./FCRO-Fair-Classification-Orthogonal-Representation/train.py` with the following command to train the FCRO model. Due to the limited computing capbaility, I just use 10,000 images for training and 5,000 images for validation.
```bash
python train.py --image_path ../ -f 0 --cond
```

3. run `./FCRO-Fair-Classification-Orthogonal-Representation/train.py` with the following command to test the trained model. I use 13,233 images to test the result, which is same as the test set in the original paper.
```bash
python train.py --test --image_path ../ -f 0 --epoch 35 --pretrained_path ../experiments/
```

# Test result

Comparing the test results of my method and the test results of the paper, I found that the AUC value of my method is lower than the AUC in the paper. This may be mainly because my training size is too small compared to the original paper. In addition, in the evaluation of fairness, I was surprised to find that most of the results obtained by my method were better than those in the original paper.
