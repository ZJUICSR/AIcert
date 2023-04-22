from matplotlib import pyplot as plt

def simple_draw(feats, save_path):
    plt.scatter(feats[:, 0], feats[:, 1], marker='o',
                color='blue', s=5, alpha=0.5)
    plt.axis('off')
    plt.savefig(save_path)
    print("[Info]Saved figure at {}".format(save_path))
    plt.clf()


def draw_contrast(feats1, feats2, save_path, vis_type):
    '''

    :param feats1:clean samples' feature (blue)
    :param feats2: poisoned ones(red)
    :param save_path: path to save fig
    :return: None
    '''
    if vis_type in ('oracle', 'svm'):
        plt.figure(figsize=(7, 5))
        # plt.xlim([-3, 3])
        plt.ylim([0, 200])

        plt.hist(feats1, bins='doane', color='blue', alpha=0.5,
                 label='Clean', edgecolor='black')
        plt.hist(feats2, bins='doane', color='red', alpha=0.5,
                 label='Poison', edgecolor='black')
    elif vis_type == 'mean_diff':
        plt.figure(figsize=(7, 5))

        plt.hist(feats1.cpu().detach().numpy(), color='blue',
                 bins='doane', alpha=0.5, label='Clean', edgecolor='black')
        plt.hist(feats2.cpu().detach().numpy(), color='red',
                 bins='doane', alpha=0.5, label='Poison', edgecolor='black')

        plt.xlabel("Distance")
        plt.ylabel("Number")
        plt.legend()
    elif vis_type == 'ss':
        plt.figure(figsize=(7, 5))
        plt.ylim([0, 300])

        plt.hist(feats1.cpu().detach().numpy(), color='blue',
                 bins='doane', alpha=0.5, label='Clean', edgecolor='black')
        plt.hist(feats2.cpu().detach().numpy(), color='red', bins=20,
                 alpha=0.5, label='Poison', edgecolor='black')

        plt.xlabel("Distance")
        plt.ylabel("Number")
        plt.legend()
    else:
        plt.scatter(feats1[:, 0],
                    feats1[:, 1], marker='o', s=5,
                    color='blue', alpha=1.0)
        plt.scatter(feats2[:, 0],
                    feats2[:, 1], marker='^', s=8,
                    color='red', alpha=0.7)

        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    print("[Info]Saved figure at {}".format(save_path))

    plt.clf()