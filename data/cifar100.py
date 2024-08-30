import torchvision

def get_dataset_cifar(path):
    label_dict = {}
    count_dict = {}
    count_dict["train"] = {}
    count_dict["val"] = {}
    train_dataset = torchvision.datasets.CIFAR100(root = path, download=True, train=True)
    val_dataset = torchvision.datasets.CIFAR100(root = path, train=False)
    for i in range(100):
        label_dict[i] = []
        count_dict["train"][i] = 0
        count_dict["val"][i] = 0
    dataset = {}
    dataset["train"] = []
    dataset["val"] = []

    for (img, lab) in iter(train_dataset):
        label_dict[lab].append(img)
        count_dict["train"][lab] += 1
    
    for (img, lab) in iter(val_dataset):
        count_dict["val"][lab] += 1

    for i in range(100):
        for img in label_dict[i]:
            dataset["train"].append({})
            dataset["train"][i]["image"] = img
            dataset["train"][i]["label"] = i
        label_dict[i] = []

    for (img, lab) in iter(val_dataset):
        label_dict[lab].append(img)


    for i in range(100):
        for img in label_dict[i]:
            dataset["val"].append({})
            dataset["val"][i]["image"] = img
            dataset["val"][i]["label"] = i

    return dataset, None, count_dict