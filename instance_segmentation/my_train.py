
from dataset import *
import sys
sys.path.append('torch_utils')


from my_utils import *
from dataset import PennFudanDataset

model_path = "model.pth"

def main():
    
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() \
         else torch.device('cpu')
    
    # our dataset has two classes only - background and person
    num_classes = 2

    # use our dataset and defined transformations
    dataset = myOwnDataset(root = sperm_dataset_root, annotation=sperm_annotations_file, transforms = get_transform(train=True))
    # dataset = PennFudanDataset(dataset_dir, get_transform(train=True))
    # test_dataset = PennFudanDataset(dataset_dir, get_transform(train=False))


    # split the dataset into train and test sets
    indices = torch.randperm(len(dataset)).tolist()
    train_dataset = torch.utils.data.Subset(dataset, indices[:])
    # test_dataset = torch.utils.data.Subset(dataset, indices[-50:])

    train_data_loader = torch.utils.data.DataLoader(
        dataset, batch_size = 3, shuffle = True, num_workers = 2, collate_fn=utils.collate_fn
    )
    # test_data_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size = 1, shuffle= False, num_workers = 2, collate_fn = utils.collate_fn
    # )

    model = get_model_instance_segmentation(num_classes = num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr = 0.005, momentum = 0.9, weight_decay = 0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma = 0.05)

    num_epochs = 20
    training_times = []
    print("starting to train")
    for epoch in range(num_epochs):
        start = time.time()
        engine.train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq = 10)
        lr_scheduler.step()
        training_times.append(time.time() - start)

        # engine.evaluate(model, test_data_loader, device = device)
        torch.save(model, model_path)
    print("training times:", training_times)

    print("THAT IS IT !")



if __name__ == '__main__':
    main()
    