def predictions(lrn, pat):
    pred = []

    for count, i in enumerate(range(pat.shape[0])):
        patches_a = pat[i, :,  :, :, :]

        test_dl = lrn.dls.test_dl(patches_a, bs=1000)
        preds, _ = lrn.get_preds(dl=test_dl)
        print(count)
        pred.append(preds.argmax(dim=1).numpy())

    return np.array(pred)