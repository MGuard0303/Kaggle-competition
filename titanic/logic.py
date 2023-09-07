import torch

import utility


def train_step(model, inputs, labels):
    model.train()
    model.optimizer.zero_grad()

    # Forward
    predictions = model(inputs)
    loss = model.loss_func(predictions, labels)

    # Backward
    loss.backward()
    model.optimizer.step()

    return loss.item(), predictions


@torch.no_grad()
def valid_step(model, inputs, labels):
    model.eval()

    predictions = model(inputs)
    loss = model.loss_func(predictions, labels)

    return loss.item(), predictions


def train(model, train_loader, epochs=20):
    utility.print_time()
    print("Start training")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        utility.print_time()
        print(f"Epoch: {epoch:02}")

        for batch_number, (inputs, labels) in enumerate(train_loader, 1):
            batch_loss, batch_prediction = train_step(model, inputs, labels)
            epoch_loss += batch_loss

        avg_train_loss = epoch_loss / batch_number
        print(f"Average Training Loss: {avg_train_loss}")

    print()
    print("Training complete")

    return model


@torch.no_grad()
def predict(model, test_loader):
    model.eval()

    prediction_list = []

    for (batch_number, inputs) in enumerate(test_loader, 1):
        inputs = inputs[0]
        batch_predictions = model(inputs)
        batch_predictions = batch_predictions.squeeze(1).round().tolist()

        for i in batch_predictions:
            prediction_list.append(i)

    return prediction_list
