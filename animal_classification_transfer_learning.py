import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import time
from PIL import Image

#Initializing parameters
epochs = 10

CATEGORY_NAMES = [
	'bear', 'chimp', 'giraffe', 'gorilla', 'llama', 'ostrich', 'porcupine', 'skunk', 'triceratops', 'zebra'
]

# Applying Transforms to the Data
image_transforms = { 
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# Load the Data
 
# Set train and valid directory paths
train_directory = 'train'
valid_directory = 'valid'
test_directory = 'test'
 
# Batch size
bs = 32
 
# Number of classes
num_classes = 10
 
# Load Data from folders
data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
    'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
}
 
# Size of Data, to be used for calculating Average Loss and Accuracy
train_data_size = len(data['train'])
valid_data_size = len(data['valid'])
test_data_size = len(data['test'])
 
# Create iterators for the Data loaded using DataLoader module
train_data_loader = torch.utils.data.DataLoader(data['train'], batch_size=bs, shuffle=True)
valid_data_loader = torch.utils.data.DataLoader(data['valid'], batch_size=bs, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(data['test'], batch_size=bs, shuffle=True)
 
# Load pretrained ResNet50 Model
model = torchvision.models.resnet50(pretrained=True)

# Freeze model parameters
for param in model.parameters():
    param.requires_grad = False
	
# Change the final layer of ResNet50 Model for Transfer Learning
fc_inputs = model.fc.in_features
 
model.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 10), 
    nn.LogSoftmax(dim=1) # For using NLLLoss()
)

# Define Optimizer and Loss Function
loss_criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training
history = []
for epoch in range(epochs):
    epoch_start = time.time()
    print("Epoch: {}/{}".format(epoch+1, epochs))
     
    # Set to training mode
    model.train()
     
    # Loss and Accuracy within the epoch
    train_loss = 0.0
    train_acc = 0.0
     
    valid_loss = 0.0
    valid_acc = 0.0
 
    for i, (inputs, labels) in enumerate(train_data_loader):
 
        inputs = inputs.to(device)
        labels = labels.to(device)
         
        # Clean existing gradients
        optimizer.zero_grad()
         
        # Forward pass - compute outputs on input data using the model
        outputs = model(inputs)
         
        # Compute loss
        loss = loss_criterion(outputs, labels)
         
        # Backpropagate the gradients
        loss.backward()
         
        # Update the parameters
        optimizer.step()
         
        # Compute the total loss for the batch and add it to train_loss
        train_loss += loss.item() * inputs.size(0)
         
        # Compute the accuracy
        ret, predictions = torch.max(outputs.data, 1)
        correct_counts = predictions.eq(labels.data.view_as(predictions))
         
        # Convert correct_counts to float and then compute the mean
        acc = torch.mean(correct_counts.type(torch.FloatTensor))
         
        # Compute total accuracy in the whole batch and add to train_acc
        train_acc += acc.item() * inputs.size(0)
         
        print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))
		
    # Validation - No gradient tracking needed
    with torch.no_grad():
     
        # Set to evaluation mode
        model.eval()
     
        # Validation loop
        for j, (inputs, labels) in enumerate(valid_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
     
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
     
            # Compute loss
            loss = loss_criterion(outputs, labels)
     
            # Compute the total loss for the batch and add it to valid_loss
            valid_loss += loss.item() * inputs.size(0)
     
            # Calculate validation accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
     
            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
     
            # Compute total accuracy in the whole batch and add to valid_acc
            valid_acc += acc.item() * inputs.size(0)
     
            print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

    # Find average training loss and training accuracy
    avg_train_loss = train_loss/train_data_size 
    avg_train_acc = train_acc/float(train_data_size)
     
    # Find average training loss and training accuracy
    avg_valid_loss = valid_loss/valid_data_size 
    avg_valid_acc = valid_acc/float(valid_data_size)
     
    history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
             
    epoch_end = time.time()
     
    print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))

# predict label of an image
def predict(model, test_image_name):
     
    transform = image_transforms['test']
 
    test_image = Image.open(test_image_name)
    # plt.imshow(test_image)
     
    test_image_tensor = transform(test_image)
 
    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
     
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(1, dim=1)
        print("Predicted class :  ", CATEGORY_NAMES[topclass.cpu().numpy()[0][0]])


# sample of testing
print("Actual class: bear");
predict(model, "./test/bear/009_0071.jpg")
print()
print("Actual class: chimp");
predict(model, "./test/chimp/038_0071.jpg")
print()
print("Actual class: giraffe");
predict(model, "./test/giraffe/084_0071.jpg")
print()
print("Actual class: gorilla");
predict(model, "./test/gorilla/090_0071.jpg")
print()
print("Actual class: llama");
predict(model, "./test/llama/134_0071.jpg")
print()
print("Actual class: ostrich");
predict(model, "./test/ostrich/151_0071.jpg")
print()
print("Actual class: porcupine");
predict(model, "./test/porcupine/164_0071.jpg")
print()
print("Actual class: skunk");
predict(model, "./test/skunk/186_0071.jpg")
print()
print("Actual class: triceratops");
predict(model, "./test/triceratops/228_0071.jpg")
print()
print("Actual class: zebra");
predict(model, "./test/zebra/250_0071.jpg")
print()