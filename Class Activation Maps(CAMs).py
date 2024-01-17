# V4: Augmentations use
# V3: Added early stopping
# V2: Added Test set
# V1: Increased functionality

from torch_snippets import T, Glob, DataLoader, optim, resize, subplots
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from torchsummary import summary
from sklearn.model_selection import train_test_split
from utils import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(time.time())

train_tfms = T.Compose([
    T.ToPILImage(),
    T.Resize(128),
    T.CenterCrop(128),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])

aug1 = T.Compose([
    T.ToPILImage(),
    T.Resize(128),
    T.CenterCrop(128),
    T.RandomAffine(5, translate=(0.01, 0.1)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])

val_tfms = T.Compose([
    T.ToPILImage(),
    T.Resize(128),
    T.CenterCrop(128),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])

id2int = {'Parasitized': 0, 'Uninfected': 1}
training_files = Glob('training_set/*/*.png')
np.random.seed(10)
np.random.shuffle(training_files)

# Load and split training set
train_files, val_files = train_test_split(training_files, test_size=0.2, random_state=1)
train_ds = Malaria(train_files, transform=train_tfms)
aug1_ds = Malaria(train_files, transform=aug1)
val_dataset = Malaria(val_files, transform=val_tfms)
train_dataset = ConcatDataset([train_ds, aug1_ds])

train_dataloader = DataLoader(train_dataset, 32, shuffle=True, collate_fn=train_ds.collate_fn)
val_dataloader = DataLoader(val_dataset, 32, shuffle=True, collate_fn=val_dataset.collate_fn)

model = MalariaClassifier().to(device)
summary(model, input_size=(3, 128, 128))
# Hyper Parameter setting
criterion = model.compute_metrics
optimizer = optim.Adam(model.parameters(), lr=1e-3)
n_epochs = 10
early_stop = 3
min_loss = torch.tensor(float('inf'))
# Lists to save metrics
train_loss = []
validation_loss = []
train_acc = []
validation_acc = []
for epoch in tqdm(range(n_epochs), desc='Epoch', unit='Epoch', position=0, leave=True):
    # Begin training mode, i.e. gradient changes
    model.train(True)
    avg_loss, avg_acc = train_per_epoch(train_dataloader, model, optimizer, criterion)
    # Save loss per epoch for later visualizations
    train_loss.append(avg_loss)
    train_acc.append(avg_acc)
    # Begin evaluation mode, no gradient changes
    model.eval()
    val_loss, val_acc = evaluation(val_dataloader, model, 'Validation')
    validation_loss.append(val_loss)
    validation_acc.append(val_acc)
    print('\n')
    info = f'''Epoch: {epoch + 1:02d}\tTrain Loss: {avg_loss:.4f}\tTrain Accuracy: {avg_acc:.3f}\t'''
    info += f'\nValidation Loss: {val_loss:.4f}\tValidation Accuracy: {val_acc:.3f}\n'
    print(info)
    # Implement early stopping
    if val_loss < min_loss:
        min_loss = val_loss
        best_epoch = epoch
        early_stopping(model, "best_model.pth", 'save')
    elif epoch - best_epoch + 1 > early_stop:
        print("Early stopping training at epoch {}, best epoch {}".format(epoch, best_epoch))
        early_stopping(model, "best_model.pth", 'restore')
        break  # terminate the training loop

# Plot training metrics
plot_metrics(train_acc, validation_acc, train_loss, validation_loss)


# Evaluate on unseen data
test_files = Glob('testing_set/*/*.png')
test_dataset = Malaria(test_files, transform=val_tfms)
test_dataloader = DataLoader(test_dataset, 256, shuffle=False, collate_fn=test_dataset.collate_fn)
test_loss, test_acc = evaluation(test_dataloader, model, 'Test')
print("Test Accuracy : {:.4f}, Test Loss : {:.4f}".format(test_acc, test_loss))
display_predictions(model, id2int, test_dataset, val_tfms)
prediction_metrics(model, test_dataloader, id2int, confusion=False)

SZ = 128
N = 10
_test_dl = DataLoader(test_dataset, batch_size=N, shuffle=True, collate_fn=test_dataset.collate_fn)
x, y, z = next(iter(_test_dl))
for i in range(N):
    image = resize(z[i], SZ)
    heatmap, pred = im2gradcam(x[i:i+1], model)
    if pred == 'Uninfected':
        continue
    heatmap = upsample_heatmap(heatmap, image, SZ)
    subplots([image, heatmap], nc=2, figsize=(5, 3), suptitle=pred)
