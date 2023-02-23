import DeepSst as ds

ds.DeepSst(dataset='mnist',
          pertube=0.05,
          gpu=2,
          save_path='.',
          modelname='LeNet',
          path='./mnist_lenet5.pth',
          m_dir='.')