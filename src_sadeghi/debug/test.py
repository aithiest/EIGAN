from data import get_loader
import numpy as np

for dataset in ['cifar100', 'yaleb', 'german', 'adult']:
    print(dataset)
    train_loader = get_loader(dataset, 500000, True)
    valid_loader = get_loader(dataset, 500000, False)

    for X, Y, S in train_loader:
        print(X.shape, Y.shape, S.shape)
        y = np.unique(Y.long().flatten().numpy())
        s = np.unique(S.long().flatten().numpy())
        print('ny:', len(y), 'ns:', len(s))

    for X, Y, S in valid_loader:
        print(X.shape, Y.shape, S.shape)
        y = np.unique(Y.long().flatten().numpy())
        s = np.unique(S.long().flatten().numpy())
        print('ny:', len(y), 'ns:', len(s))
