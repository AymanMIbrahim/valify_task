from train.helpers.dataloaders import build_dataloaders
from train.helpers.evaluate import evaluate_model
from train.helpers.export import export_best_model_to_onnx
from train.helpers.trainer import fit
from train.models import SpoofFormer


def main():
    train_loader, test_loader = build_dataloaders()
    model = SpoofFormer()

    fit(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
    )

    evaluate_model()
    export_info = export_best_model_to_onnx()
    print("ONNX export completed.")
    print(export_info)


if __name__ == "__main__":
    main()