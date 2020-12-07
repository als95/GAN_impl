import torch
from fid_score.fid_score import FidScore

device = torch.device('cuda:0')
batch_size = 1


def cal_fid_score(name):
    paths = ['images/' + name + '/origin', 'images/' + name + '/gen']
    fid = FidScore(paths, device, batch_size)
    score = fid.calculate_fid_score()

    return score


# names = ['acgan', 'bgan', 'began', 'cgan', 'ebgan', 'lsgan', 'sgan', 'relativistic_gan', 'wgan_gp']
names = ['dagan']
scores = []

for name in names:
    scores.append(cal_fid_score(name))

for i in range(len(names)):
    print('%s\'s fid-score : %f' % (names[i], scores[i]))

