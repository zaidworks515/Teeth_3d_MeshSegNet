import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from Mesh_dataset import *
from meshsegnet import *
from losses_and_metrics_for_mesh import *
import utils
import pandas as pd
import gc

if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(utils.get_avail_gpu())
    print(f"Using device: {device}")

    # Paths and Parameters
    train_list = './train_list_1.csv'
    val_list = './val_list_1.csv'
    model_path = './models/'
    model_name = 'Teeth Segmentation (Trial)'
    checkpoint_name = './latest_checkpoint.tar'  # Save checkpoint in the root directory for simplicity

    num_classes = 15  # Update to 15 to match the range of label values
    num_channels = 15  # number of features
    num_epochs = 100
    num_workers = 0
    train_batch_size = 2 # Lowered for debugging
    val_batch_size = 2
    num_batches_to_print = 5  # More frequent printing for debugging

    # Create 'models' directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)

    # Load Datasets
    try:
        print("Loading training dataset...")
        training_dataset = Mesh_Dataset(data_list_path=train_list, num_classes=num_classes, patch_size=6000)
        print("Loading validation dataset...")
        val_dataset = Mesh_Dataset(data_list_path=val_list, num_classes=num_classes, patch_size=6000)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

    train_loader = DataLoader(dataset=training_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    # Initialize Model
    model = MeshSegNet(num_classes=num_classes, num_channels=num_channels, with_dropout=True, dropout_p=0.5).to(device)
    opt = optim.Adam(model.parameters(), amsgrad=True)

    # Track metrics
    losses, mdsc, msen, mppv = [], [], [], []
    val_losses, val_mdsc, val_msen, val_mppv = [], [], [], []
    best_val_dsc = 0.0

    # Enable cuDNN if available for faster performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    print('Starting training...')
    class_weights = torch.ones(num_classes).to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss, running_mdsc, running_msen, running_mppv = 0.0, 0.0, 0.0, 0.0
        loss_epoch, mdsc_epoch, msen_epoch, mppv_epoch = 0.0, 0.0, 0.0, 0.0

        for i_batch, batched_sample in enumerate(train_loader):
            inputs = batched_sample['cells'].to(device, dtype=torch.float)
            labels = batched_sample['labels'].to(device, dtype=torch.long)
            A_S = batched_sample['A_S'].to(device, dtype=torch.float)
            A_L = batched_sample['A_L'].to(device, dtype=torch.float)

            # Adjust labels to be zero-indexed
            labels = labels - 2  # Assuming original labels range from 2 to 15

            print(f"Labels shape: {labels.shape}")  # Debugging line
            print(f"Labels dtype: {labels.dtype}")  # Debugging line
            unique_labels = torch.unique(labels)
            print(f"Unique labels: {unique_labels}")

            print(f"number of classes: {len(unique_labels)}")

            one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes)

            opt.zero_grad()
            outputs = model(inputs, A_S, A_L)

            # Calculate Losses and Metrics
            loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
            dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
            sen = weighting_SEN(outputs, one_hot_labels, class_weights)
            ppv = weighting_PPV(outputs, one_hot_labels, class_weights)

            loss.backward()
            opt.step()

            # Update metrics
            running_loss += loss.item()
            running_mdsc += dsc.item()
            running_msen += sen.item()
            running_mppv += ppv.item()
            loss_epoch += loss.item()
            mdsc_epoch += dsc.item()
            msen_epoch += sen.item()
            mppv_epoch += ppv.item()

            # Print intermediate results
            if i_batch % num_batches_to_print == num_batches_to_print - 1:
                print(f'[Epoch: {epoch + 1}/{num_epochs}, Batch: {i_batch + 1}/{len(train_loader)}] '
                      f'loss: {running_loss / num_batches_to_print}, dsc: {running_mdsc / num_batches_to_print}, '
                      f'sen: {running_msen / num_batches_to_print}, ppv: {running_mppv / num_batches_to_print}')
                running_loss, running_mdsc, running_msen, running_mppv = 0.0, 0.0, 0.0, 0.0

        # Append epoch metrics
        losses.append(loss_epoch / len(train_loader))
        mdsc.append(mdsc_epoch / len(train_loader))
        msen.append(msen_epoch / len(train_loader))
        mppv.append(mppv_epoch / len(train_loader))

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss_epoch, val_mdsc_epoch, val_msen_epoch, val_mppv_epoch = 0.0, 0.0, 0.0, 0.0

            for i_batch, batched_val_sample in enumerate(val_loader):
                inputs = batched_val_sample['cells'].to(device, dtype=torch.float)
                labels = batched_val_sample['labels'].to(device, dtype=torch.long)
                A_S = batched_val_sample['A_S'].to(device, dtype=torch.float)
                A_L = batched_val_sample['A_L'].to(device, dtype=torch.float)

                # Adjust labels to be zero-indexed
                labels = labels - 2  # Assuming original labels range from 2 to 15
                one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes)

                outputs = model(inputs, A_S, A_L)

                # Validation Losses and Metrics
                loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
                dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
                sen = weighting_SEN(outputs, one_hot_labels, class_weights)
                ppv = weighting_PPV(outputs, one_hot_labels, class_weights)

                val_loss_epoch += loss.item()
                val_mdsc_epoch += dsc.item()
                val_msen_epoch += sen.item()
                val_mppv_epoch += ppv.item()

                if i_batch % num_batches_to_print == num_batches_to_print - 1:
                    print(f'[Epoch: {epoch + 1}/{num_epochs}, Val Batch: {i_batch + 1}/{len(val_loader)}] '
                          f'val_loss: {val_loss_epoch / num_batches_to_print}, val_dsc: {val_mdsc_epoch / num_batches_to_print}, '
                          f'val_sen: {val_msen_epoch / num_batches_to_print}, val_ppv: {val_mppv_epoch / num_batches_to_print}')

            # Append validation metrics
            val_losses.append(val_loss_epoch / len(val_loader))
            val_mdsc.append(val_mdsc_epoch / len(val_loader))
            val_msen.append(val_msen_epoch / len(val_loader))
            val_mppv.append(val_mppv_epoch / len(val_loader))

        # Print epoch results
        print(f'*\nEpoch: {epoch + 1}/{num_epochs}, loss: {losses[-1]}, dsc: {mdsc[-1]}, '
              f'sen: {msen[-1]}, ppv: {mppv[-1]}\n'
              f'val_loss: {val_losses[-1]}, val_dsc: {val_mdsc[-1]}, val_sen: {val_msen[-1]}, val_ppv: {val_mppv[-1]}\n*')

        # Save the checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'losses': losses,
            'mdsc': mdsc,
            'msen': msen,
            'mppv': mppv,
            'val_losses': val_losses,
            'val_mdsc': val_mdsc,
            'val_msen': val_msen,
            'val_mppv': val_mppv
        }, checkpoint_name)

        # Save the best model
        if best_val_dsc < val_mdsc[-1]:
            best_val_dsc = val_mdsc[-1]
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'losses': losses,
                'mdsc': mdsc,
                'msen': msen,
                'mppv': mppv,
                'val_losses': val_losses,
                'val_mdsc': val_mdsc,
                'val_msen': val_msen,
                'val_mppv': val_mppv
            }, model_path + f'{model_name}_best.tar')

        # Save metrics to CSV
        pd_dict = {
            'train_loss': losses,
            'train_dsc': mdsc,
            'train_sen': msen,
            'train_ppv': mppv,
            'val_loss': val_losses,
            'val_dsc': val_mdsc,
            'val_sen': val_msen,
            'val_ppv': val_mppv
        }
        df = pd.DataFrame(pd_dict)
        df.to_csv(model_path + model_name + '_stats.csv', index=False)

