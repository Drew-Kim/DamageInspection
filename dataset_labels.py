import fiftyone as fo

#Create dataset, add types and import data
dataset = fo.Dataset.from_dir(
    dataset_dir=r"C:\Users\morad\Downloads\dataset3",
    dataset_type=fo.types.ImageDirectory,
    name = "BoxDataset",
    overwrite=True
    )

#Loads dataset to Voxl
dataset = fo.load_dataset("BoxDataset")

#For exporting the dataset into certain classes (damaged_box, undamaged_box, opened_box)
damaged_view = dataset.filter_labels(
    "ground_truth",
    fo.ViewField("label") == "damaged_box"
)

damaged_view.export(
    export_dir="export_damaged_box3",
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
)

opened_view = dataset.filter_labels(
    "ground_truth",
    fo.ViewField("label") == "opened_box"
)

opened_view.export(
    export_dir="export_opened_box3",
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
)

undamaged_view = dataset.filter_labels(
    "ground_truth",
    fo.ViewField("label") == "undamaged_box"
)

undamaged_view.export(
    export_dir="export_undamaged_box3",
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
)

#Launches the fiftyone app to visualize the dataset
session = fo.launch_app(dataset, port=5151)

session.wait()