import joblib, json
import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score, roc_curve
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

from utils.cnn3d_tools import EarlyStopping, FocalLoss, TverskyLoss, save_checkpoint, load_checkpoint
import os

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out

class PatchedR3D18(nn.Module):
    def __init__(self, num_classes=2):
        super(PatchedR3D18, self).__init__()
        self.in_planes = 64

        self.stem = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        )

        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes)
            )

        layers = []
        layers.append(BasicBlock3D(self.in_planes, planes, stride, downsample))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def load_pretrained_weights_safely(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # If the checkpoint has 'state_dict' key (sometimes happens), unwrap it
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    model_dict = model.state_dict()
    pretrained_dict = {}

    for k, v in checkpoint.items():
        # Remove 'module.' prefix if present
        if k.startswith('module.'):
            k = k[7:]  # remove 'module.' prefix

        if k in model_dict and v.shape == model_dict[k].shape:
            pretrained_dict[k] = v
        else:
            print(f"Skipping layer: {k} (shape mismatch or not found)")

    # Load matched weights
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print(f"Loaded {len(pretrained_dict)} layers from pretrained model.")
    return model

def get_model(device, num_classes=2, pretrained_path=None,
              unfreeze_layers=['layer2', 'layer3', 'layer4', 'fc']):
    """
    Builds patched ResNet-18, loads weights if path given,
    then freezes everything except the layers in unfreeze_layers list.
    """
    model = PatchedR3D18(num_classes=num_classes)

    if pretrained_path is not None:
        model = load_pretrained_weights_safely(model, pretrained_path)

    # Freeze initially
    for p in model.parameters(): p.requires_grad = False
    
    # Selectively un-freeze
    for name, module in model.named_children():
        if name in unfreeze_layers:
            for p in module.parameters(): p.requires_grad = True

    return model.to(device)

def train(
        device,
        model,
        train_loader,
        val_loader=None,
        extra_epochs=5,
        fine_tune_lr=1e-4,
        plot_dir='plots',
        early_stop_patience=7,
        resume_checkpoint=True,
        load_ckpt_path='models/checkpoint.pth',
        save_ckpt_path='models/checkpoint.pth'
        ):

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=fine_tune_lr,
        weight_decay=1e-4
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)

    # Loss: Combine CE, Focal, and Tversky
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    foc = FocalLoss(gamma=4.0)
    tv  = TverskyLoss(alpha=0.3, beta=0.7)

    def mixed_loss(out, y):
        return 0.3 * ce(out, y) + 0.4 * foc(out, y) + 0.3 * tv(out, y)

    # Resume checkpoint
    start_epoch = 0
    best_loss = float('inf')
    if resume_checkpoint and os.path.exists(load_ckpt_path):
        model, optimizer, scheduler, start_epoch, best_loss = load_checkpoint(
            model, optimizer, scheduler, path=load_ckpt_path
        )
        start_epoch += 1
        print(f"Resuming at epoch {start_epoch}")

    early_stopper = EarlyStopping(patience=early_stop_patience, verbose=True)
    early_stopper.best_loss = best_loss

    # 5. Training loop
    model.train()
    train_losses, val_losses, lrs = [], [], []
    final_epoch = start_epoch + extra_epochs

    for epoch in range(start_epoch, final_epoch):
        running_test_loss = 0.0
        for scans, labels in train_loader:
            scans, labels = scans.to(device), labels.to(device)
            optimizer.zero_grad()
            tloss = mixed_loss(model(scans), labels)
            tloss.backward()
            optimizer.step()
            running_test_loss += tloss.item()

        epoch_loss_t = running_test_loss / len(train_loader)

        running_val_loss = 0.0
        for scans, labels in val_loader:
            scans, labels = scans.to(device), labels.to(device)
            optimizer.zero_grad()
            vloss = mixed_loss(model(scans), labels)
            vloss.backward()
            optimizer.step()
            running_val_loss += vloss.item()

        epoch_loss_v = running_val_loss / len(val_loader)

        scheduler.step(epoch_loss_v)

        if epoch_loss_v < best_loss:
            best_loss = epoch_loss_v

        print(f"Epoch [{epoch+1}/{final_epoch}]  Train loss: {tloss:.4f}    Val loss: {vloss:.4f}")
        train_losses.append(epoch_loss_t)
        val_losses.append(epoch_loss_v)
        lrs.append(optimizer.param_groups[0]['lr'])

        save_checkpoint(model, optimizer, scheduler, epoch, best_loss, path=save_ckpt_path)

        early_stopper(epoch_loss_v)
        if early_stopper.early_stop:
            print("Early stopping triggered!")
            break

    # 6. Curves
    plt.figure(); 
    plt.plot(train_losses, 'o-'); 
    plt.title('Train loss'); 
    plt.savefig(os.path.join(plot_dir,'tloss_curve.png')); 
    plt.close()

    plt.figure(); 
    plt.plot(val_losses, 'o-'); 
    plt.title('Val loss'); 
    plt.savefig(os.path.join(plot_dir,'vloss_curve.png')); 
    plt.close()

    plt.figure(); 
    plt.plot(lrs, 'o-');         
    plt.title('LR');   
    plt.savefig(os.path.join(plot_dir,'lr_curve.png'));   
    plt.close()                        

def evaluate_and_plot(model, device, test_loader, plot_dir):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    low_confidence_samples = []
    all_logits = []


    with torch.no_grad():
        for i, (scans, labels) in enumerate(test_loader):
            scans = scans.to(device)
            labels = labels.to(device)

            outputs = model(scans)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_logits.append(outputs[:,1].item())         # store logit for class 1

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs[:,1].cpu())

            confidence = probs.max().item()
            low_confidence_samples.append((i, preds.item(), labels.item(), confidence))

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    print("Classification Report:")
    print(classification_report(all_labels, all_preds))

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', 'Schizophrenic'])
    disp.plot(cmap='Blues')
    plt.savefig(os.path.join(plot_dir, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = roc_auc_score(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'roc_curve.png'), bbox_inches='tight')
    plt.close()

    print("Confusion matrix, ROC curve saved.")

    # Save low confidence cases
    low_confidence_sorted = sorted(low_confidence_samples, key=lambda x: x[3])[:5]
    print("\nLowest confidence predictions:")
    for idx, pred, label, conf in low_confidence_sorted:
        print(f"Sample #{idx}: Predicted={pred}, True={label}, Confidence={conf:.2f}")
        