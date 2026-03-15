import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE,
    DATA_PATH,
    DROP_COLUMNS,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
    NUM_EPOCHS,
    PATIENCE,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    USE_POS_WEIGHT,
    VALID_SIZE,
)
from src.data_preparation import (
    inspect_data,
    load_data,
    prepare_target,
    prepare_targets_for_torch,
    preprocess_data,
    split_data,
    split_features_target,
)
from src.dataset import BankDataset
from src.model import MLP
from src.train import fit


def main():
    df = load_data(DATA_PATH)
    inspect_data(df)

    df = prepare_target(df, TARGET_COLUMN)

    x, y = split_features_target(df, TARGET_COLUMN, DROP_COLUMNS)

    x_train, x_valid, x_test, y_train, y_valid, y_test = split_data(
        x, y, TEST_SIZE, VALID_SIZE, RANDOM_STATE
    )

    x_train_processed, x_valid_processed, x_test_processed, preprocessor = (
        preprocess_data(x_train, x_valid, x_test)
    )

    y_train_prepared, y_valid_prepared, y_test_prepared = prepare_targets_for_torch(
        y_train, y_valid, y_test
    )

    train_dataset = BankDataset(x_train_processed, y_train_prepared)
    valid_dataset = BankDataset(x_valid_processed, y_valid_prepared)
    test_dataset = BankDataset(x_test_processed, y_test_prepared)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = x_train_processed.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Используемое устройство:", device)

    model = MLP(input_dim).to(device)

    if USE_POS_WEIGHT:
        negative_count = (y_train == 0).sum()
        positive_count = (y_train == 1).sum()

        pos_weight_value = float(negative_count / positive_count)
        pos_weight = torch.tensor([pos_weight_value], device=device)
        print("pos_weight:", pos_weight_value)
    else:
        pos_weight = None

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = fit(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        model_save_path=MODEL_SAVE_PATH,
    )

    print("\nОбучение завершено.")
    print("Количество завершённых эпох:", len(history["train_loss"]))


if __name__ == "__main__":
    main()
